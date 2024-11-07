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

NS_MODEL* create_model(NS_NEURON** input_neurons, uint64_t n_input)
{
	NS_MODEL* model = malloc(sizeof(NS_MODEL));
	if (!model)
		return 0;
	model->input_neurons = input_neurons;
	for (uint64_t i = 0; i < n_input; ++i)
		// no parent means it's an input neuron
		create_synapse(0, model->input_neurons[i]);
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
}

float neuron_forward(NS_NEURON* neuron)
{
	NS_SYNAPSE* parent_synapse = neuron->parents;
	// If the neuron is an input layer neuron, just return its value
	if (!parent_synapse->parent && parent_synapse->input_value)
		return parent_synapse->input_value;
	float inputs = 0.f;
	for (uint64_t i = 0; i < neuron->n_parents; ++i)
		inputs += neuron_forward(neuron->parents[i]->parent) * neuron->parents[i]->weight;
	return neuron->function(inputs);
}

NS_ARRAY* get_final_children(NS_NEURON* neuron)
{
	NS_ARRAY* final_neurons = ns_array_create();
	if (neuron->n_children == 0)
		ns_array_append(final_neurons, neuron);
	else if(neuron->children)
		return get_final_children(neuron->children[0]);
	return final_neurons;
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
