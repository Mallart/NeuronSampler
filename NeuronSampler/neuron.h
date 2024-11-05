#pragma once
#include "neuron_functions.h";
#define BIND_CONST_LAYERS(parent_layer, child_layer) bulk_bind_layers(parent_layer, size_of_neuron_array(parent_layer), child_layer, size_of_neuron_array(child_layer))

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
	NS_SYNAPSE** parents;
	NS_SYNAPSE** children;
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
// forward propagation of a neuron
float neuron_forward(NS_NEURON* neuron);
// returns the final child from the given neuron.
NS_NEURON* get_final_child(NS_NEURON* neuron);
// sets model input values
void set_input_values(NS_MODEL* model, float* input_values, uint64_t n_inputs);
// binds two layers, connecting every single neuron from layer2 to every single neuron of layer2.
void bulk_bind_layers(NS_NEURON** parent_layer, uint64_t n_parent_layer_neurons, NS_NEURON** child_layer, uint64_t n_child_layer_neurons);
// returns the size of a const neuron array (layer).
// caution: doesn't work with dynamically created layers.
uint64_t size_of_neuron_array(NS_NEURON* layer[]);