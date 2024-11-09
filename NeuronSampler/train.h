#pragma once
//#include <threads.h>
#include "libs.h"
#include "neuron.h"
// includes multi threading purposes and training functions.

// epoch is the number of trainings.
void train_model(NS_MODEL* model, NS_TARGET* target, uint64_t epoch, double precision);

// saves the model in the given file.
void save_model_state(NS_MODEL* model, FILE* stream);
// reads the model saved state in given file
NS_MODEL* read_model_state(FILE* stream);

clock_t benchmark(void* (*f)(void));
clock_t benchmark_training(void (*training)(NS_MODEL*, NS_TARGET*, uint64_t), NS_MODEL* nsm, NS_TARGET* nst, uint64_t epoch);