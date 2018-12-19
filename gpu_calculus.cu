#include "gpu_calculus.h"
#include "calculus.h"
#include "math.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define BLOCKDIM 32
#define MAXARRAYSIZE 130000000 // supoe GPU RAM < 1GB

cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess)
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    return result;
}


// realiza a reducao do vetor g_idata e devolve o resultado em g_odata
__global__ void reduction(double *g_idata, long long n, double *g_odata) {
    extern __shared__ double sdata[];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        sdata[tid] = g_idata[i];

        __syncthreads();
        // do reduction in shared mem
        unsigned int s = blockDim.x / 2;
        while (s > 0) {
            if (tid < s)
                sdata[tid] += sdata[tid + s];
            __syncthreads();
            s /= 2;
        }

        // write result for this block to global mem
        if (tid == 0) g_odata[blockIdx.x] = sdata[0];
    }
}


// recebe um vetor x e calcula f(x) inplace
__global__ void calculate_fx(double *x, long long n, long long k, long long M) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = (sin((2 * M + 1) * M_PI * x[i]) * cos(2 * M_PI * k * x[i])) / sin(M_PI * x[i]);
}


// recebe um vetor f(x) e calcula f(x)^2 inplace
__global__ void calculate_fx_2(double *fx, long long n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) fx[i] = fx[i] * fx[i];
}


// cria um vetor de tamanho n com x aleatório entre (0, 0.5]
static double *create_random_x(long long n) {
    double *x = create_empty_array(n);
    for (int i = 0; i < n; i++)
        x[i] = get_random_x();
    return x;
}


// recebe N, k, M e calcula <f> e <f^2>
void gpu_get_f(long long N, long long k, long long M, double *f, double *f2) {

    *f = *f2 = 0;

    for (long long task = N; task > 0; task -= MAXARRAYSIZE) {

        long long n = (task < MAXARRAYSIZE) ? task : MAXARRAYSIZE;
        long long grid_dim = (n + BLOCKDIM-1) / BLOCKDIM;
        double *x_h = create_random_x(n);
        double *result_h = (double *) malloc(grid_dim * sizeof(double));

        double *x_d, *result_d;
        checkCuda( cudaMalloc((void **) &x_d, n * sizeof(double)) );
        checkCuda( cudaMalloc((void **) &result_d, grid_dim * sizeof(double)) );

        // calcula f(x) no device
        checkCuda( cudaMemcpy( x_d, x_h, n * sizeof(double), cudaMemcpyHostToDevice) );
        calculate_fx<<<grid_dim, BLOCKDIM>>>(x_d, n, k, M);

        // reduz f(x) no device e termina no host
        reduction <<<grid_dim, BLOCKDIM, BLOCKDIM>>>(x_d, n, result_d);
        checkCuda( cudaMemcpy( result_h, result_d, grid_dim * sizeof(double), cudaMemcpyDeviceToHost) );;
        for (int i = 0; i < grid_dim; i++)
            *f += result_h[i];

        // calcula f(x)^2 no device
        calculate_fx_2<<<grid_dim, BLOCKDIM>>>(x_d, n);

        // reduz f(x)^2 no device e termina no host
        reduction <<<grid_dim, BLOCKDIM, BLOCKDIM>>>(x_d, n, result_d);
        checkCuda( cudaMemcpy( result_h, result_d, grid_dim * sizeof(double), cudaMemcpyDeviceToHost) );
        for (int i = 0; i < grid_dim; i++)
            *f2 += result_h[i];

        // limpeza
        checkCuda( cudaFree(x_d) );
        checkCuda( cudaFree(result_d) );
        free(x_h);
        free(result_h);
    }

    *f /= N;  // encontra <f>
    *f2 /= N; // encontra <f^2>
}


// recebe N, k, M, calcula os dois resultados da integral de Monte Carlo e devolve o tempo de execucao
double gpu_monte_carlo(long long N, long long k, long long M, double *result_sum, double *result_sub) {
    double f, f2, start, finish;
    start = omp_get_wtime();
    gpu_get_f(N, k, M, &f, &f2);
    *result_sum = monte_carlo_sum(f, f2, N);
    *result_sub = monte_carlo_sub(f, f2, N);
    finish = omp_get_wtime();
    return finish - start;
}
