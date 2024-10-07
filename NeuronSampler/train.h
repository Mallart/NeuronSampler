#pragma once
#include <threads.h>
#include "libs.h"
#include "neuron.h"
// includes multi threading purposes and training functions.

void train_model(NS_MODEL* model, uint64_t n_iterations);