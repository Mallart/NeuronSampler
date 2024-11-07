#pragma once
#include "data_structures.h"
#include "neuron_functions.h";
// returns the size of a const neuron array (layer).
// caution: doesn't work with dynamically created layers.
#define CONST_LAYER_SIZE(layer) sizeof(layer) / sizeof(NS_NEURON*)
#define BIND_CONST_LAYERS(parent_layer, child_layer) bulk_bind_layers(parent_layer, sizeof(parent_layer) / sizeof(NS_NEURON*), child_layer, sizeof(child_layer) / sizeof(NS_NEURON*))
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
} NS_MODEL;

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
// returns the final children from the given neuron.
// CAUTION: currently broken, doesn't work properly.
NS_ARRAY* get_final_children(NS_NEURON* neuron);
// sets model input values
void set_input_values(NS_MODEL* model, float* input_values, uint64_t n_inputs);
// binds two layers, connecting every single neuron from layer2 to every single neuron of layer2.
void bulk_bind_layers(NS_NEURON** parent_layer, uint64_t n_parent_layer_neurons, NS_NEURON** child_layer, uint64_t n_child_layer_neurons);
// sets the activation function of an array of neurons
void layer_set_function(float (*function)(float), NS_NEURON** layer, uint64_t n_neurons);