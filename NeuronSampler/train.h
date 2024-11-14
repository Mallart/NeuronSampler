#pragma once
//#include <threads.h>
#include "libs.h"
#include "neuron.h"
// includes multi threading purposes and training functions.

// epoch is the number of trainings.
void train_model(NS_MODEL* model, NS_TARGET* target, uint64_t epoch, double precision);
// train the model using several targets.
void mass_train_model(NS_MODEL* model, NS_DATASET* dataset, uint64_t epoch, double learning_rate);
// use the model with new input values
void model_query(NS_MODEL* model, NS_TARGET* input);
// saves the model in the given file.
void save_model_state(NS_MODEL* model, FILE* stream);
// reads the model saved state in given file
NS_MODEL* read_model_state(FILE* stream);

clock_t benchmark(void* (*f)(void));
clock_t benchmark_training(void (*training)(NS_MODEL*, NS_TARGET*, uint64_t, double), NS_MODEL* nsm, NS_TARGET* nst, uint64_t epoch, double learning_rate);
clock_t benchmark_model_creation(NS_MODEL* (*model_creation_function)(void));