#pragma once
#include <threads.h>
#include "libs.h"
#include "neuron.h"
// includes multi threading purposes and training functions.

// epoch is the number of trainings.
void train_model(NS_MODEL* model, uint64_t epoch);