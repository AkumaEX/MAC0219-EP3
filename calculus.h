#ifndef CALCULUS_H
#define CALCULUS_H

double get_random_x();

double monte_carlo_sum(double f, double f_2, long long N);

double monte_carlo_sub(double f, double f_2, long long N);

double *create_empty_array(long long N);

#endif //CALCULUS_H
