int*
cond_distr(int m, int w, int n_topics, float beta, float alpha, 
            int* my_nzw, int* nmz, int num_words)
{
        // num_words = num words for this processor
        // n_topics = number of topics in total
        // nz = my_nzw summed over w
        // int* nz = malloc(n_topics * sizeof(int));
        int nz[10];
        for (int i = 0; i < n_topics; i++)
        {
            for (int j = 0; j < num_words; j++) 
            {
                if (nz[i])
                    nz[i] += my_nzw[i * num_words + j];
                else
                    nz[i] = my_nzw[i * num_words + j];
            }
        }

        int vocab_size = num_words;
        int left[10];
        int right[10];

        for (int i = 0; i < n_topics; i++)
        {
            int j;
        }

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
       __global __read_only int* nzw, 
       __global __read_only int* nmz,
       __global __write_only int* gpu_pnz, 
       int p, int n_topics, float alpha, float beta)
{
    // Global position of output pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    return x;
}

int
sample_index(int p) 
{
    //Sample from the Multinomial distribution and return the sample index.
    //return np.random.multinomial(1,p).argmax()
    int j;
}
