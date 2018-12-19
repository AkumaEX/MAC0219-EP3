#include "load_balance.h"
#include "gpu_calculus.h"
#include "omp_calculus.h"
#include "seq_calculus.h"
#include <stdio.h>
#include <mpi.h>
#include <omp.h>

#define N_TRAIN 500


// devolve a taxa de crescimento da funcao linear dados dois pontos
double get_ratio(double y, double y0, long long x, long long x0) {
    return (y - y0) / (x - x0);
}


// recebe dados de tempo de execucao e devolve a quantidade de amostras que a GPU devera executar dado N
long long predict_gpu_n(double gpu_y0, double gpu_y, double omp_y0, double omp_y, long long x0, long long x, long long N) {
    double gpu_ratio = get_ratio(gpu_y, gpu_y0, x, x0);
    double omp_ratio = get_ratio(omp_y, omp_y0, x, x0);
    return (long long)(N * omp_ratio / (gpu_ratio + omp_ratio));
}


// calcula o numero de amostras para GPU (gpu_N) e OMP (omp_N)
void get_n_samples(long long N, long long k, long long M, long long *gpu_N, long long *omp_N, int world_rank) {
    *gpu_N = *omp_N = 0;
    long long x0 = 1;
    long long x = 100000;
    double gpu_y0, gpu_y, omp_y0, omp_y, result_sum, result_sub;
    gpu_y0 = gpu_y = omp_y0 = omp_y = 0.0;

    if (world_rank == 0) {
        printf("treinando ... \n");
        for (int i = 0; i < N_TRAIN; i++) {
            gpu_y0 += gpu_monte_carlo(x0, k, M, &result_sum, &result_sub)  / N_TRAIN;
            gpu_y  += gpu_monte_carlo( x, k, M, &result_sum, &result_sub)  / N_TRAIN;
        }
        MPI_Recv(&omp_y0, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&omp_y, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        *gpu_N = predict_gpu_n(gpu_y0, gpu_y, omp_y0, omp_y, x0, x, N);
        *omp_N = N - *gpu_N;
        printf("gpu_N: %lld, omp_N: %lld\n", *gpu_N, *omp_N);
        printf("--------------------------------------------------------------------------------\n");

    } else if (world_rank == 1) {
        for (int i = 0; i < N_TRAIN; i++) {
            omp_y0 += omp_monte_carlo(x0, k, M, &result_sum, &result_sub) / N_TRAIN;;
            omp_y  += omp_monte_carlo( x, k, M, &result_sum, &result_sub) / N_TRAIN;;
        }
        MPI_Send(&omp_y0, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&omp_y , 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
}
