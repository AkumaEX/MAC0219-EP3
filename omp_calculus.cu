#include "omp_calculus.h"
#include "calculus.h"
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#define M_PI 3.14159265358979323846
#define MAXARRAYSIZE 260000000 // supoe RAM principal < 2GB


// realiza a reducao do vetor idata
static double reduction(double *idata, long long n) {
    double sum = 0;
#pragma omp parallel for reduction (+:sum)
    for (int i = 0; i < n; i++)
        sum += idata[i];
    return sum;
}


// recebe um vetor de x de tamanho n e devolve f(x) inplace
static double *calculate_fx(double *x, long long n, long long k, long long M) {
#pragma omp parallel for
    for (int i = 0; i < n; i++)
        x[i] = (sin((2 * M + 1) * M_PI * x[i]) * cos(2 * M_PI * k * x[i])) / sin(M_PI * x[i]);
    return x;
}


// recebe um vetor de f(x) de tamanho n e devolve f(x)^2 inplace
static double *calculate_fx_2(double *fx, long long n) {
#pragma omp parallel for
    for (int i = 0; i < n; i++)
        fx[i] = fx[i] * fx[i];
    return fx;
}


// cria um vetor x aleatorio de tamanho n
static double *create_random_x(long long n) {
    double *x = create_empty_array(n);
#pragma omp parallel for
    for (int i = 0; i < n; i++)
        x[i] = get_random_x();
    return x;
}


// recebe N, k, M e realiza o calculo de <f> e <f^2>
void omp_get_f(long long N, long long k, long long M, double *f, double *f2) {

    *f = *f2 = 0;

    for (long long task = N; task > 0; task -= MAXARRAYSIZE) {
        long long n = (task < MAXARRAYSIZE) ? task : MAXARRAYSIZE;
        double *x = create_random_x(n);
        double *fx = calculate_fx(x, n, k, M);
        *f += reduction(fx, n);
        double *fx_2 = calculate_fx_2(fx, n);
        *f2 += reduction(fx_2, n);
        free(x);
    }

    *f /= N;
    *f2 /= N;
}


// recebe N, k, M, calcula os dois resultados da integral de Monte Carlo e devolve o tempo de execucao
double omp_monte_carlo(long long N, long long k, long long M, double *result_sum, double *result_sub) {
    double f, f2, start, finish;
    start = omp_get_wtime();
    omp_get_f(N, k, M, &f, &f2);
    *result_sum = monte_carlo_sum(f, f2, N);
    *result_sub = monte_carlo_sub(f, f2, N);
    finish = omp_get_wtime();
    return finish - start;
}