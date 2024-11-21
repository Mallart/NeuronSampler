#pragma once
#include "data_structures.h"
#include "neuron_functions.h"
// returns the size of a const neuron array (layer).
// caution: doesn't work with dynamically created layers.
#define CONST_LAYER_SIZE(layer) CONST_ARRAY_SIZE(NS_NEURON*, layer)
#define BIND_CONST_LAYERS(parent_layer, child_layer) bulk_bind_layers(parent_layer, sizeof(parent_layer) / sizeof(NS_NEURON*), child_layer, sizeof(child_layer) / sizeof(NS_NEURON*))

static uint64_t NEURON_NUMBER = 1;

struct NS_NEURON;
struct NS_SYNAPSE;

typedef uint32_t numerator;

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
	// numerator n_parents;
	// numerator n_children;
	// NS_SYNAPSE** parents;
	// NS_SYNAPSE** children;
	NS_SYNAPSE_ARRAY* parents;
	NS_SYNAPSE_ARRAY* children;
	double (*function)(double); // activation function
	// value set only once, to avoid the neuron to recompute the same value several times
	// hence reducing the stack size while processing huge models
	double value;
	double bias;
	double delta;
	// temporary and will be removed in a future version
	uint64_t id;
	NS_FLAG role;
} NS_NEURON;

// target version of model for a neural network.
// "Given this input, I want this output".
typedef struct NS_TARGET
{
	numerator n_inputs, n_outputs;
	double* inputs;
	double* outputs;
} NS_TARGET;

// A model consists of several neurons.
typedef struct NS_MODEL
{
	// model's input neurons
	NS_NEURON** input_neurons;
	NS_NEURON** output_neurons;
	numerator n_input_neurons;
	numerator n_output_neurons;
} NS_MODEL;

// creates a new synapse between two neurons (binds them automatically)
NS_SYNAPSE* create_synapse(NS_NEURON* parent, NS_NEURON* child);
// unbinds two neurons
void destroy_synapse(NS_SYNAPSE* synapse);
// creates a new neuron
NS_NEURON* create_neuron();
// creates a new array of n neurons
NS_NEURON** create_layer(uint64_t n);
// creates a new model from an array of input neurons
NS_MODEL* create_model(NS_NEURON** input_neurons, uint64_t n_input, NS_NEURON** output_neurons, uint64_t n_output);
// reset an existing neuron.
void init_neuron(NS_NEURON* neuron);
// forward propagation of a neuron (neuron parameter is the last / output neuron)
double neuron_forward(NS_NEURON* neuron);
// backward propagation to adjust the weights and biases (neuron parameter is the last / output neuron)
void neuron_backwards(NS_NEURON* neuron, double target, double learning_rate);
// sets model input values
void set_input_values(NS_MODEL* model, float* input_values, uint64_t n_inputs);
// binds two layers, connecting every single neuron from layer2 to every single neuron of layer2.
void bulk_bind_layers(NS_NEURON** parent_layer, uint64_t n_parent_layer_neurons, NS_NEURON** child_layer, uint64_t n_child_layer_neurons);
// sets the activation function of an array of neurons
void layer_set_function(double (*function)(double), NS_NEURON** layer, uint64_t n_neurons);

// will replace all input values with the given ones.
// prepares the model to be trained.
void model_feed_values(NS_MODEL* model, NS_TARGET* target);

void layer_add_current_neurons(NS_LAYER* layer, NS_NEURON* neuron);
NS_LAYER* model_get_all_neurons(NS_MODEL* model);

void delete_synapse(NS_SYNAPSE* synapse);
void delete_neuron(NS_NEURON* neuron);
void delete_layer(NS_LAYER* layer);
void delete_model(NS_MODEL* model);

#pragma region SERIALIZATION

char* serialize_neuron(NS_NEURON* neuron);
NS_NEURON* deserialize_neuron(char* buffer);

char* serialize_synpase(NS_SYNAPSE* synapse);
NS_SYNAPSE* deserialize_synapse(char* buffer);

char* serialize_model(NS_MODEL* model);
NS_MODEL* deserialize_model(char* buffer);
#pragma endregion