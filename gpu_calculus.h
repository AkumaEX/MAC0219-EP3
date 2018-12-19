#ifndef GPU_CALCULUS_H
#define GPU_CALCULUS_H

void gpu_get_f(long long N, long long k, long long M, double *f, double *f2);

double gpu_monte_carlo(long long N, long long k, long long M, double *result_sum, double *result_sub);

#endif //GPU_CALCULUS_H
