#pragma once
#include "libs.h"
#include "data_structures.h"

#pragma region R_NEURON_FUNCTIONS


double relu(double x);
double sigmoid(double x);
// returns the raw value of this neuron (no edition)
double raw(double x);
double heaviside(double x);

static const double (*NEURON_FUNCTIONS[])(double) = 
{ 
    relu, 
    sigmoid, 
    raw,
    heaviside,
}; 
enum NEURON_ACTIVATION_FUNCTIONS {
    RELU, 
    SIGMOID, 
    RAW,
    HEAVISIDE,
};

#pragma endregion For everything related to the forward pass / propagation of results

#pragma region R_ERROR_FUNCTIONS
// to compute the error rate and learning gradients.
double average_quadratic_error(uint64_t n_results, double* result, double* expected);

#pragma region R_DERIVATIVES
double d_relu(double x);
double d_sigmoid(double x);
double d_raw(double x);
// compute the derivative of any function in one X point.
double d_function(double(*f)(double), double x);

#pragma endregion

#pragma region To compute errors, gradients, and make the model learn.