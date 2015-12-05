int*
cond_distr()
{

}

__kernel void
sample(__global __read_only int* topics, 
       __global __read_only int* matrix, 
       __global __read_only int* nzw, 
       __global __read_only int p,
       __global __write_only int* gpu_pnz, 
       int* nmz, float alpha, float beta)
{
    // Global position of output pixel
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    float c_real, c_imag;
    float z_real, z_imag;
    int iter;

    if ((x < w) && (y < h)) {
        // YOUR CODE HERE
        c_real = coords_real[x + y * w];
        c_imag = coords_imag[x + y * w];
        z_real = 0;
        z_imag = 0;

        for (iter = 0; iter < max_iter; iter++){

            if ((z_real * z_real + z_imag * z_imag) > 4)
                break;
            z_real_temp = z_real;
            z_real = c_real + (z_real * z_real) - (z_imag * z_imag);
            z_imag = 2 * z_real_temp * z_imag + c_imag;
        }

        out_counts[x + y * w] = iter;
    }
}
