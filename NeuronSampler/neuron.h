#pragma once
#include "libs.h";

struct NS_NEURON;
struct NS_SYNAPSE;

typedef struct NS_SYNAPSE
{
	struct NS_NEURON* parent;
	struct NS_NEURON* child;
	// synapse's weight
	float weight;
} NS_SYNAPSE;

typedef struct NS_NEURON
{
	uint64_t n_parents;
	uint64_t n_children;
	struct NS_SYNAPSE** parents;
	struct NS_SYNAPSE** children;
	float (*function)(float); // activation function
} NS_NEURON;

// appends an element on an array of pointers and returns the new array
// WARNING: can destroy the old array if there's not enough free space to add an element
void* array_append(void** array, uint64_t array_size, void* element);
void* array_append_no_duplicate(void** array, uint64_t array_size, void* element);
// replaces an element with 0.
void array_remove(void** array, uint64_t array_size, void* element);
// returns true if the specified element exist in array
bool array_exists(void** array, uint64_t array_size, void* element);

// creates a new synapse between two neurons (binds them automatically)
NS_SYNAPSE* create_synapse(NS_NEURON* parent, NS_NEURON* child);
// unbinds two neurons
void destroy_synapse(NS_SYNAPSE* synapse);
// creates a new neuron
NS_NEURON* create_neuron();
// reset an existing neuron.
void init_neuron(NS_NEURON* neuron);
