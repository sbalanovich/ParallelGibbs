int searchsorted(__global float* arr, int length, float value){
    /*
    Bisection search (c.f. numpy.searchsorted)
    Find the index into sorted array `arr` of length `length` such that, if
    `value` were inserted before the index, the order of `arr` would be
    preserved. From https://github.com/ariddell/lda/blob/develop/lda/_lda.pyx
    */
    int imin = 0;
    int imax = length;
    int imid = 0;
    // printf("START\n");
    while (imin < imax) {
        imid = imin + ((imax - imin) >> 2);
        // printf("%d %d %d %f %f\n", imin, imid, imax, value, arr[imid]);
        if (value > arr[imid])
            imin = imid + 1;
        else
            imax = imid;
    }

    return imin;
}

int cond_distr(int m, int w, int n_topics, float beta, float alpha, float randi,
            __local int* nzw, __local int* nz, __local int* nmz, __global float* dist_sum, 
            int n_words, int n_docs) {
        // num_words = num words for this processor
        // n_topics = number of topics in total

        // int vocab_size = num_words;
        // int left[10];
        // int right[10];

        // // left = (nzw[:,w] + beta) / (nz + beta * vocab_size)
        // // right = (self.nmz[m,:] + self.alpha)

        // // int* pnz = left * right;
        // int pnz[10];
        // int pnz_sum = 0;
        // for(int i = 0; i < n_topics; i++) {
        //     pnz_sum += pnz[i];
        // }
        // // normalize to obtain probabilities
        // for(int i = 0; i < n_topics; i++) {
        //     pnz[i] /= pnz_sum;
        // }
        // return pnz;


        // algorithm for sampling from https://github.com/ariddell/lda/blob/develop/lda/_lda.pyx
        float dist_cum = 0;
        // double* dist_sum = (double*) malloc(n_topics * sizeof(double));
        for (int k =0; k < n_topics; k++) {
            dist_cum += (nzw[k * n_words + w] + beta) / 
                        (nz[k] + beta * n_words) * 
                        (nmz[m * n_topics + k] + alpha);
            dist_sum[k] = dist_cum;
        }
        
        return searchsorted(dist_sum, n_topics, randi * dist_cum);
        
}

__kernel void
sample(__global int* topics, 
       __global int* matrix, 
       __global int* nzw, 
       __global int* nmz,
       __global int* nz, 
       __global int* nm,
       __global float* rands, 
       __global float* dist_sum, 
       __local int *topic_buffer,
       __local int *nmz_buffer, __local int *nm_buffer,
       __local int *nzw_buffer, __local int *nz_buffer,
       __local int* nz_copy_buffer,
       int n_topics, int n_words, 
       int n_docs, float alpha, float beta)
{
    // Pull in all the ids
    size_t local_id = get_local_id(0);
    size_t global_id = get_global_id(0);
    size_t group_id = get_group_id(0);
    size_t global_sz = get_global_size(0);
    size_t local_sz = get_local_size(0);

    global_sz = (global_sz <= 0 ) ? 1 : global_sz;
    int k_words = ceil((float) n_words / global_sz);
    int k_docs = ceil((float) n_docs / global_sz);
    // printf("%d %d %d %d %d\n", local_id, global_id, group_id, k_words, k_docs);

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int t = 0; t < n_topics; t++) {
        nz_copy_buffer[t] = nz[t];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Load the relevant topics to a local buffer
    for (int d = 0; d < k_docs; d++) {
        int doc = k_docs * global_id + d;
        for (int w = 0; w < k_words; w++) {
            int word = k_words * global_id + w;
            topic_buffer[d * k_words + w] = topics[doc * n_words + word];
            if ((d * k_words + w) > (k_docs * k_words)){
                printf("Fail1\n");
            }
            if ((doc * n_words + word) > (n_docs * n_words)){
                printf("Fail2\n");
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Load the relevant nmzs to a local buffer
    int nmzs_sz = k_words;
    for (int n = 0; n < nmzs_sz; n++) {
        int doc = k_docs * global_id + n;
        for (int topic = 0; topic < n_topics; topic++) {
            nmz_buffer[n * n_topics + topic] = nmz[doc * n_topics + topic];
            if ((n * n_topics + topic) > (k_docs * n_topics)){
                printf("Fail3\n");
            }
            if ((doc * n_topics + topic) > (n_docs * n_topics)){
                printf("Fail4\n");
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Load the relevant nms to a local buffer
    int nms_sz = k_docs;
    for (int n = 0; n < nms_sz; n++) {
        int doc = k_docs * global_id + n;
        nm_buffer[n] = nm[doc];
        if ((n) > (k_docs)){
            printf("Fail5\n");
        }
        if ((doc) > (n_docs)){
            printf("Fail6\n");
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Load the relevant nzws to a local buffer
    int nzws_sz = k_words;
    for (int topic = 0; topic < n_topics; topic++) {
        for (int n = 0; n < nzws_sz; n++) {
            int word = k_words * global_id + n;
            nzw_buffer[topic * k_words + n] = nzw[topic * n_words + word];
            if ((topic * k_words + n) > (k_words * n_topics)){
                printf("Fail7\n");
            }
            if ((topic * n_words + word) > (n_words * n_topics)){
                printf("Fail8\n");
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Load the relevant nzs to a local buffer
    int nzs_sz = n_topics;
    for (int n = 0; n < nzs_sz; n++) {
        nz_buffer[n] = nz[n];
        if ((n) > (n_topics)){
            printf("Fail9\n");
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int m = 0; m < k_docs; m++)
    {
        for (int w = 0; w < k_words; w++)
        {
            int z = topic_buffer[m * k_words + w];
            // printf("# %d\n", z);
            nmz_buffer[m * n_topics + z] -= 1;
            nm_buffer[m] -= 1;
            nzw_buffer[z * k_words + w] -= 1;
            nz_buffer[z] -= 1;
            // printf("#%d %d %d %d %d#\n", 
            //     z, nmz_buffer[m * n_topics + z], nm_buffer[m],
            //     nzw_buffer[z * k_words + w], nz_buffer[z]);

            float randi = rands[m * k_words + w];
            // printf("#######%f\n", randi);
            z = cond_distr(m, w, n_topics, beta, alpha, 
                           randi, nzw_buffer, nz_buffer, nmz_buffer, 
                           dist_sum, k_words, k_docs);
            // printf("%d\n", z);

            nmz_buffer[m * n_topics + z] += 1;
            nm_buffer[m] += 1;
            nzw_buffer[z * k_words + w] += 1;
            nz_buffer[z] += 1;
            topic_buffer[m * k_words + w] = z;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int topic = 0; topic < n_topics; topic++) {
        for (int n = 0; n < nzws_sz; n++) {
            int word = k_words * global_id + n;
            // printf("##%d %d\n", n, nz_buffer[n]);
            nzw[topic * n_words + word] += (nzw_buffer[topic * k_words + n] - nzw[topic * n_words + word]);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int n = 0; n < nzs_sz; n++) {
        nz[n] += (nz_buffer[n] - nz_copy_buffer[n]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int d = 0; d < k_docs; d++) {
        int doc = k_docs * global_id + d;
        for (int w = 0; w < k_words; w++) {
            int word = k_words * global_id + w;
            topics[doc * n_words + word] = topic_buffer[d * k_words + w];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Load the relevant nmzs to a local buffer
    for (int n = 0; n < nmzs_sz; n++) {
        int doc = k_docs * global_id + n;
        for (int topic = 0; topic < n_topics; topic++) {
            nmz[doc * n_topics + topic] = nmz_buffer[n * n_topics + topic];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Load the relevant nms to a local buffer
    for (int n = 0; n < nms_sz; n++) {
        int doc = k_docs * global_id + n;
        nm[doc] = nm_buffer[n];
    }

    barrier(CLK_LOCAL_MEM_FENCE);



    //update global values
    // new nzw
    //for each 
    // nzw[z, w]
    // global nzw += local nzw - global nzw
}
