int*
cond_distr(int m, int w, int n_topics, float beta, float alpha, 
            int* my_nzw, int* nz, int* nmz, int num_words)
{
        // num_words = num words for this processor
        // n_topics = number of topics in total

        int vocab_size = num_words;
        int left[10];
        int right[10];

        // left = (nzw[:,w] + beta) / (nz + beta * vocab_size)
        // right = (self.nmz[m,:] + self.alpha)

        // int* pnz = left * right;
        int pnz[10];
        int pnz_sum = 0;
        for(int i = 0; i < n_topics; i++) {
            pnz_sum += pnz[i];
        }
        // normalize to obtain probabilities
        for(int i = 0; i < n_topics; i++) {
            pnz[i] /= pnz_sum;
        }
        return pnz;
}

__kernel void
sample(__global __read_only int* topics, 
       __global __read_only int* matrix, 
       __global __read_only int* my_nzw, 
       __global __read_only int* nmz,
       __global __write_only int* gpu_pnz, 
       int p, int n_topics, int n_words, 
       int n_docs, float alpha, float beta)
{
    // Build NZ array
    // n_words = num words for this processor
    // n_topics = number of topics in total
    // nz = my_nzw summed over w
    // int* nz = malloc(n_topics * sizeof(int));
    int nz[10];
    for (int i = 0; i < n_topics; i++)
    {
        for (int j = 0; j < n_words; j++) 
        {
            if (nz[i])
                nz[i] += my_nzw[i * n_words + j];
            else
                nz[i] = my_nzw[i * n_words + j];
        }
    }

    size_t local_id = get_local_id(0);
    size_t global_id = get_global_id(0);
    size_t group_id = get_group_id(0);
    long global_sz = get_global_size(0);
    long local_sz = get_local_size(0);

    int k_words = ceil((float) n_words / global_sz);
    int k_docs = ceil((float) n_docs / global_sz);
    printf("%d %d %d %d %d\n", local_id, global_id, group_id, k_words, k_docs);

    // thread with global_id 0 should go through docs 0..k_docs-1
    //                             and through words 0..k_words-1
    // thread with global_id 1 should go through docs k_docs..2k_docs-1
    //                             and through words k_words-1..2k_words-1
    // thread with global_id 2 should go through docs 2k_docs..3k_docs-1
    //                             and through words 2k_words..3k_words-1
    // 
    // Be careful that each thread stays in bounds, both relative to
    // size of x (i.e., N), and the range it's assigned to sum.
    for (int m = k_docs * global_id; m < k_docs * (global_id + 1) && m < n_docs; m++) {
        w_idx = 0;
        for (int w = k_words * global_id; w < k_words * (global_id + 1) && w < n_words; w++) { 
            // m = i
            ;; // Do stuff per word slice here
            w_idx++;
        }
    }

    return 1;
    // for m in docs_by_processor[p]:
    //     for i, w in enumerate(word_indices(matrix[m, :])):
    //         z = topics[(m,i)]
    //         nmz[m,z] -= 1
    //         nm[m] -= 1
    //         local_nzw[p][z,w] -= 1
    //         local_nz[p][z] -= 1

    //         p_z = conditional_distribution(m, w, p, nmz, local_nzw, local_nz, alpha, beta)
    //         if not np.isclose(np.sum(p_z), 1.):
    //             print p_z
    //         z = sample_index(p_z)

    //         nmz[m,z] += 1
    //         nm[m] += 1
    //         local_nzw[p][z,w] += 1
    //         local_nz[p][z] += 1
    //         topics[m,i] = z
}

int
sample_index(int p) 
{
    //Sample from the Multinomial distribution and return the sample index.
    //return np.random.multinomial(1,p).argmax()
    int j;
}
