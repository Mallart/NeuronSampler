#include "neuron_functions.h"

float relu(float x)
{
    return max(0, x);
}

float sigmoid(float x)
{
    return 1.f / (1 + exp(-x));
}

float raw(float x)
{
    return x;
}
