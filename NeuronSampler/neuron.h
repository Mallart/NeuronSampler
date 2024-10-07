#pragma once
#include "neuron_functions.h";

struct NS_NEURON;
struct NS_SYNAPSE;


typedef struct NS_SYNAPSE
{
	struct NS_NEURON* parent;
	struct NS_NEURON* child;
	// synapse's weight
	union
	{
		// when the bound neuron is an input one, this value should be set.
		float input_value;
		// when the bound neuron isn't an input one, this value should be set.
		float weight;
	};
} NS_SYNAPSE;

typedef struct NS_NEURON
{
	uint64_t n_parents;
	uint64_t n_children;
	struct NS_SYNAPSE** parents;
	struct NS_SYNAPSE** children;
	float bias;
	float (*function)(float); // activation function
} NS_NEURON;

// A model consists of several neurons.
typedef struct NS_MODEL
{
	// model's input neurons
	NS_NEURON** input_neurons;
	// model's output neurons. should be read-only.
	NS_NEURON** output_neurons;
} NS_MODEL;

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
// creates a new model from an array of input neurons
NS_MODEL* create_model(NS_NEURON** input_neurons, uint64_t n_input);
// reset an existing neuron.
void init_neuron(NS_NEURON* neuron);

float neuron_forward(NS_NEURON* neuron);
// returns the final child from the given neuron.
NS_NEURON* get_final_child(NS_NEURON* neuron);

void set_input_values(NS_MODEL* model, float* input_values, uint64_t n_inputs);