#ifndef OMP_CALCULUS_H
#define OMP_CALCULUS_H

void omp_get_f(long long N, long long k, long long M, double *f, double *f2);

double omp_monte_carlo(long long N, long long k, long long M, double *result_sum, double *result_sub);

#endif //OMP_CALCULUS_H
