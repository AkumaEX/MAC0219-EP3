#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <math.h>
#include <time.h>
#include "calculus.h"
#include "gpu_calculus.h"
#include "omp_calculus.h"
#include "seq_calculus.h"
#include "load_balance.h"


// imprime os resultados na tela
void print_results(long long k, long long M, double result_sum, double result_sub) {
    double result = 0;
    if (llabs(k) <= llabs(M)) result = (M >= 0) ? 1 : -1;
    printf("Erro no calculo com a soma: %lf\n", fabs(result - result_sum));
    printf("Erro no calculo com a subtracao: %lf\n", fabs(result - result_sub));
    printf("--------------------------------------------------------------------------------\n");
}


// executa a integracao de Monte Carlo usando CUDA + OpenMP
void perform_monte_carlo_hibrid(long long N, long long k, long long M, long long gpu_N, long long omp_N, int world_rank) {
    double f, f2, gpu_f, gpu_f2, omp_f, omp_f2, result_sum, result_sub, start, finish, elapsed;
    if (world_rank == 0) {
        start = omp_get_wtime();
        MPI_Send(&omp_N, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        gpu_get_f(gpu_N, k, M, &gpu_f, &gpu_f2);
        MPI_Recv(&omp_f, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&omp_f2, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        f = (gpu_f*gpu_N + omp_f*omp_N) / N;
        f2 = (gpu_f2*gpu_N + omp_f2*omp_N) / N;

        result_sum = monte_carlo_sum(f, f2, N);
        result_sub = monte_carlo_sub(f, f2, N);

        finish = omp_get_wtime();
        elapsed = finish - start;
        printf("Tempo com balanceamento de carga em segundos: %lf\n", elapsed);
        print_results(k, M, result_sum, result_sub);

    } else if (world_rank == 1) {
        MPI_Recv(&omp_N, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        omp_get_f(omp_N, k, M, &omp_f, &omp_f2);
        MPI_Send(&omp_f, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&omp_f2, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
}


// executa a integracao de Monte Carlo com CUDA
void perform_monte_carlo_gpu(long long N, long long k, long long M, int world_rank) {
    if (world_rank == 0) {
        double result_sum, result_sub, elapsed;
        elapsed = gpu_monte_carlo(N, k, M, &result_sum, &result_sub);
        printf("Tempo na GPU com uma thread na CPU em segundos: %lf\n", elapsed);
        print_results(k, M, result_sum, result_sub);
    }
}


// executa a integracao de Monte Carlo com OpenMP
void perform_monte_carlo_omp(long long N, long long k, long long M, int world_rank) {
    if (world_rank == 0) {
        double result_sum, result_sub, elapsed;
        elapsed = omp_monte_carlo(N, k, M, &result_sum, &result_sub);
        printf("Tempo na CPU com %d threads em segundos: %lf\n", omp_get_max_threads(), elapsed);
        print_results(k, M, result_sum, result_sub);
    }
}


// executa a integracao de Monte Carlo Sequencial
void perform_monte_carlo_seq(long long N, long long k, long long M, int world_rank) {
    if (world_rank == 0) {
        double result_sum, result_sub, elapsed;
        elapsed = seq_monte_carlo(N, k, M, &result_sum, &result_sub);
        printf("Tempo sequencial em segundos: %lf\n", elapsed);
        print_results(k, M, result_sum, result_sub);
    }
}


int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("Use: %s <N> <k> <M>\n", argv[0]);
        exit(3);
    }
    srand((unsigned int)time(NULL));
    long long N = atoll(argv[1]);
    long long k = atoll(argv[2]);
    long long M = atoll(argv[3]);
    long long gpu_N, omp_N;
    int world_rank, world_size;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_size < 2) {
        fprintf(stderr, "O numero de processos deve ser maior do que 1 para %s\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    get_n_samples(N, k, M, &gpu_N, &omp_N, world_rank);
    perform_monte_carlo_hibrid(N, k, M, gpu_N, omp_N, world_rank);
    perform_monte_carlo_gpu(N, k, M, world_rank);
    perform_monte_carlo_omp(N, k, M, world_rank);
    perform_monte_carlo_seq(N, k, M, world_rank);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
