#include "neuron_functions.h"

double relu(double x)
{
    return max(0, x);
}

double sigmoid(double x)
{
    return 1.f / (1 + exp(-x));
}

double raw(double x)
{
    return x;
}

double heaviside(double x)
{
    return x >= 0.0;
}

double average_quadratic_error(uint64_t n_results, double* result, double* expected)
{
    double _sum = 0;
    for (uint64_t i = 0; i < n_results; ++i)
        _sum += pow(result - expected, 2);
    return _sum / n_results;
}

double d_relu(double x)
{
    return x > 0 ? 1 : 0;
}

double d_sigmoid(double x)
{
    double _s = sigmoid(x);
    return _s * sigmoid(1.0 - _s);
}

double d_raw(double x)
{
    return 1;
}

double d_function(double(*f)(double), double x)
{
    if (f == raw)
        return d_raw(x);
    if (f == sigmoid)
        return d_sigmoid(x);
    if (f == relu)
        return d_relu(x);
    if (f == tanh)
        return 1 - pow(exp(x) - exp(-x), 2) / pow(exp(x) + exp(-x), 2);
    return (f(x + NS_EPSILON) - f(x)) / NS_EPSILON;
}
