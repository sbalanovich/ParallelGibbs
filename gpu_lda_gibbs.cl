
int cond_distr(int m, int w, int n_topics, float beta, float alpha, 
            int* my_nzw, int* nz, int* nmz, int num_words, int n_docs) {
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


        //algorithm for sampling from https://github.com/ariddell/lda/blob/develop/lda/_lda.pyx
        float dist_cum = 0;
        for k in range(n_topics) {

            dist_cum += (nzw[k * n_topics + w] + beta) / (nz[k] + beta * num_words) * (nmz[m * n_docs + k] + alpha[k]);
            dist_sum[k] = dist_cum;
        }

        return searchsorted(dist_sum, n_topics, rand() * dist_cum);
        
}


int searchsorted(double* arr, int length, double value){
    """Bisection search (c.f. numpy.searchsorted)
    Find the index into sorted array `arr` of length `length` such that, if
    `value` were inserted before the index, the order of `arr` would be
    preserved. From https://github.com/ariddell/lda/blob/develop/lda/_lda.pyx
    """
    int imin, imax, imid;
    imin = 0;
    imax = length;
    while (imin < imax) {
        imid = imin + ((imax - imin) >> 2);
        if (value > arr[imid])
            imin = imid + 1;
        else
            imax = imid;
    }
    return imin;
}

__kernel void
sample(__global __read_only int* topics, 
       __global __read_only int* matrix, 
       __global __read_only int* nzw, 
       __global __read_only int* nmz,
       __global __read_only int* nz, 
       __global __read_only int* nm,
       __global __write_only int* gpu_pnz, 
       __local int *topic_buffer,
       __local int *nmz_buffer, __local int *nm_buffer,
       __local int *nzw_buffer, __local int *nz_buffer,
       int p, int n_topics, int n_words, 
       int n_docs, float alpha, float beta)
{
    // Pull in all the ids
    size_t local_id = get_local_id(0);
    size_t global_id = get_global_id(0);
    size_t group_id = get_group_id(0);
    long global_sz = get_global_size(0);
    long local_sz = get_local_size(0);

    int k_words = ceil((float) n_words / global_sz);
    int k_docs = ceil((float) n_docs / global_sz);
    printf("%d %d %d %d %d\n", local_id, global_id, group_id, k_words, k_docs);

    // Load the relevant topics to a local buffer
    int topics_sz = n_docs * n_words;
    for (int t = 0; t < topics_sz; t++) {
        int doc = k_docs * global_id + t;
        int word = k_words * global_id + t;
        topic_buffer[t] = topics[doc * n_docs + word];
    }

    // barrier(CLK_LOCAL_MEM_FENCE);

    // Load the relevant nmzs to a local buffer
    int nmzs_sz = n_docs;
    for (int n = 0; n < nmzs_sz; n++) {
        int doc = k_docs * global_id + n;
        for (int topic = 0; topic < n_topics; topic++) {
            nmz_buffer[n] = nmz[doc * n_docs + topic];
        }
    }

    // barrier(CLK_LOCAL_MEM_FENCE);

    // Load the relevant nms to a local buffer
    int nms_sz = n_docs;
    for (int n = 0; n < nms_sz; n++) {
        int doc = k_docs * global_id + n;
        nm_buffer[n] = nm[doc * n_docs];
    }

    // barrier(CLK_LOCAL_MEM_FENCE);

    // Load the relevant nzws to a local buffer
    int nzws_sz = n_words;
    for (int topic = 0; topic < n_topics; topic++) {
        for (int n = 0; n < nzws_sz; n++) {
            int word = k_words * global_id + n;
            nzw_buffer[n] = nzw[topic * n_topics + word];
        }
    }

    // barrier(CLK_LOCAL_MEM_FENCE);

    // Load the relevant nzs to a local buffer
    int nzs_sz = n_topics;
    for (int n = 0; n < nzs_sz; n++) {
        nz_buffer[n] = nz[n];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int m = 0; m < k_docs; m++)
    {
        for (int w = 0; w < k_words; w++)
        {
            int z = topic_buffer[m * k_docs + w];
            nmz_buffer[m * k_docs + z] -= 1;
            nm_buffer[m] -= 1;
            nzw_buffer[z * n_topics + w] -= 1;
            nz_buffer[z] -= 1;

            // z = cond_distr(inputs);

            nmz_buffer[m * k_docs + z] += 1;
            nm_buffer[m] += 1;
            nzw_buffer[z * n_topics + w] += 1;
            nz_buffer[z] += 1;
            topic_buffer[m * k_docs + w] = z;
        }
    }

    // barrier(CLK_LOCAL_MEM_FENCE);

    return 1;
}
