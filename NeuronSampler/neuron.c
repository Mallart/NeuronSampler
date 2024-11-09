#include "neuron.h"


NS_SYNAPSE* create_synapse(NS_NEURON* parent, NS_NEURON* child)
{
	NS_SYNAPSE* synapse = malloc(sizeof(NS_SYNAPSE));
	if (!synapse)
		return 0;
	synapse->parent = parent;
	synapse->child = child;
	synapse->input_value = 0;
	synapse->weight = 0;
	
	if (parent && !array_exists(parent->children, parent->n_children, child))
	{
		parent->children = array_append(parent->children, parent->n_children, synapse);
		parent->n_children++;
	}
	
	if (child && !array_exists(child->parents, child->n_parents, parent))
	{
		child->parents = array_append(child->parents, child->n_parents, synapse);
		child->n_parents++;
	}

	return synapse;
}

void destroy_synapse(NS_SYNAPSE* synapse)
{
	NS_NEURON* parent = synapse->parent;
	NS_NEURON* child = synapse->child;

	if (!array_exists(parent->children, parent->n_children, child))
	{
		array_remove(parent->children, parent->n_children, synapse);
		parent->n_children--;
	}
	
	if (!array_exists(child->parents, child->n_parents, parent))
	{
		array_remove(child->parents, child->n_parents, synapse);
		parent->n_parents--;
	}

	free(synapse);
}

NS_NEURON* create_neuron()
{
	NS_NEURON* neuron = calloc(1, sizeof(NS_NEURON));
	if (!neuron)
		return 0;
	init_neuron(neuron);
	return neuron;
}

NS_NEURON** create_layer(uint64_t n)
{
	NS_NEURON** neurons = calloc(n, sizeof(NS_NEURON*));
	if (!neurons)
		return 0;
	for (uint64_t i = 0; i < n; ++i)
		neurons[i] = create_neuron();
	return neurons;
}

NS_MODEL* create_model(NS_NEURON** input_neurons, uint64_t n_input, NS_NEURON** output, uint64_t n_output)
{
	NS_MODEL* model = malloc(sizeof(NS_MODEL));
	if (!model)
		return 0;
	model->n_input_neurons = n_input;
	model->input_neurons = input_neurons;
	model->output_neurons = output;
	model->n_output_neurons = n_output;
	for (uint64_t i = 0; i < n_input; ++i)
	{
		// no parent means it's an input neuron
		create_synapse(0, model->input_neurons[i]);
		model->input_neurons[i]->role = ROLE_INPUT;

	}
	for (uint64_t i = 0; i < n_output; ++i)
		model->output_neurons[i]->role |= ROLE_OUTPUT;
	return model;
}

void init_neuron(NS_NEURON* neuron)
{
	if(neuron->parents)
		free(neuron->parents);
	if(neuron->children)
		free(neuron->children);
	neuron->parents = malloc(sizeof(NS_SYNAPSE**));
	neuron->children = malloc(sizeof(NS_SYNAPSE**));
	neuron->n_parents = 0;
	neuron->n_children = 0;
	neuron->function = 0;
	neuron->bias = 0;
	neuron->value = 0;
	neuron->role = 0;
}

float neuron_forward(NS_NEURON* neuron)
{
	NS_SYNAPSE* parent_synapse = neuron->parents;
	// If the neuron is an input layer neuron, just return its value
	if (!parent_synapse->parent && parent_synapse->input_value)
		return parent_synapse->input_value;
	double inputs = 0.f;
	for (uint64_t i = 0; i < neuron->n_parents; ++i)
		if (neuron->parents[i]->parent && neuron->parents[i]->parent->value)
			inputs += neuron->parents[i]->parent->value * neuron->parents[i]->weight;
		else if(neuron->parents[i]->parent)
			inputs += neuron_forward(neuron->parents[i]->parent) * neuron->parents[i]->weight;
	double _val = neuron->function(inputs + neuron->bias);
	neuron->value = _val;
	return _val;
}

void neuron_backwards(NS_NEURON* neuron, double target, double learning_rate)
{
	// TODO | FIX this is buggy
	if (neuron->role == ROLE_INPUT);
		return;
	// it's not an input neuron
	double error = neuron->value - target;
	neuron->delta = error * d_function(neuron->function, neuron->value);
	for (uint64_t i = 0; i < neuron->n_parents; ++i)
	{
		if (!neuron->parents[i]->parent)
			continue;
		neuron->parents[i]->parent->delta += neuron->delta * neuron->parents[i]->weight;
		neuron->parents[i]->weight -= learning_rate * neuron->delta * neuron->parents[i]->parent->value;
		neuron_backwards(neuron->parents[i], REPLACE_THIS_VALUE_WITH_WORKING_ONE, learning_rate);
	}
	neuron->bias -= learning_rate * neuron->delta;
}

void set_input_values(NS_MODEL* model, float* input_values, uint64_t n_inputs)
{
	for (uint64_t i = 0; i < n_inputs; ++i)
		model->input_neurons[i]->parents[0]->input_value = input_values[i];
}

void bulk_bind_layers(NS_NEURON** parent_layer, uint64_t n_parent_layer_neurons, NS_NEURON** child_layer, uint64_t n_child_layer_neurons)
{
	for (uint64_t i = 0; i < n_parent_layer_neurons; ++i)
		for (uint64_t ii = 0; ii < n_child_layer_neurons; ++ii)
			create_synapse(parent_layer[i], child_layer[ii]);
}

void layer_set_function(float (*function)(float), NS_NEURON** layer, uint64_t n_neurons)
{
	for (uint64_t i = 0; i < n_neurons; ++i)
		layer[i]->function = function;
}

char* serialize_neuron(NS_NEURON* neuron)
{
	return 0;
}

NS_NEURON* deserialize_neuron(char* buffer)
{
	return 0;
}

void model_feed_values(NS_MODEL* model, NS_TARGET* target)
{
	// constantly gets removed for no reason, so I put a save here
	// every variable seem corrupted atp
	NS_NEURON_ARRAY* ns_output = ns_array_create_from_buffer(model->output_neurons, model->n_output_neurons);
	uint64_t n_values = min(model->n_input_neurons, target->n_inputs);
	for (uint64_t i = 0; i < n_values; ++i)
	{
		model->input_neurons[i]->role |= LIT_STATE;
		model->input_neurons[i]->value = target->inputs[i];
	}
	model->output_neurons = array_create_from_ns_array(ns_output);
}
