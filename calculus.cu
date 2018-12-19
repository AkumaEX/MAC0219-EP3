#include "calculus.h"
#include <math.h>
#include <stdlib.h>

double get_random_x() {
    return ((double)(rand() + 1)) / 2.0 / ((double)RAND_MAX) ;
}

double monte_carlo_sum(double f, double f_2, long long N) {
    return f + sqrt((f_2 - (f * f)) / N );
}

double monte_carlo_sub(double f, double f_2, long long N) {
    return f - sqrt((f_2 - (f * f)) / N );
}

double *create_empty_array(long long N) {
    return (double *) calloc(N, sizeof(double));
}
