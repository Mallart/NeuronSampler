#include "neuron.h"


NS_SYNAPSE* create_synapse(NS_NEURON* parent, NS_NEURON* child)
{
	srand(time(0));
	NS_SYNAPSE* synapse = malloc(sizeof(NS_SYNAPSE));
	if (!synapse)
		return 0;
	synapse->parent = parent;
	synapse->child = child;
	if (parent && !(parent->role & ROLE_INPUT | parent->role & ROLE_OUTPUT | parent->role & ROLE_HIDDEN))
		parent->role = ROLE_HIDDEN;
	if (child && !(child->role & ROLE_INPUT | child->role & ROLE_OUTPUT | child->role & ROLE_HIDDEN))
		child->role = ROLE_HIDDEN;
	synapse->weight = fmod(((double)rand()) / 1000000, 1.f);
	
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
		model->input_neurons[i]->role |= ROLE_INPUT;

	}
	for (uint64_t i = 0; i < n_output; ++i)
		model->output_neurons[i]->role |= ROLE_OUTPUT;
	layer_set_function(raw, model->input_neurons, model->n_input_neurons);

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
	neuron->id = NEURON_NUMBER++;
}

double neuron_forward(NS_NEURON* neuron)
{
	if (!neuron->role)
		return .0f;
	neuron->role |= LIT_STATE;
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
	if (neuron->role & ROLE_INPUT)
		return;
	// it's not an input neuron
	double error = 0;
	if(neuron->role & ROLE_OUTPUT)
		error = neuron->value - target;
	else
	{
		for (uint64_t i = 0; i < neuron->n_children; ++i)
			error += neuron->children[i]->weight * neuron->children[i]->child->delta;
		error = d_function(neuron->function, neuron->value) * error;
	}
	neuron->delta = error;
	neuron->bias -= learning_rate * neuron->delta;
	for (uint64_t i = 0; i < neuron->n_parents; ++i)
	{
		if (!neuron->parents[i]->parent)
			continue;
		neuron->parents[i]->weight -= learning_rate * neuron->delta * neuron->parents[i]->parent->value;
		neuron_backwards(neuron->parents[i]->parent, target, learning_rate);
	}
	// shuts down the neuron
	if (neuron->role & LIT_STATE)
		neuron->role &= ~LIT_STATE;;
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
	// constantly gets removed for no reason, so I put a save here (thanks MSVC)
	// every variable seem corrupted atp
	NS_NEURON* ns_output = model->output_neurons[0];
	uint64_t n_values = min(model->n_input_neurons, target->n_inputs);
	for (uint64_t i = 0; i < n_values; ++i)
	{
		model->input_neurons[i]->role |= LIT_STATE;
		model->input_neurons[i]->value = target->inputs[i];
		model->input_neurons[i]->parents[0]->input_value = target->inputs[i];
	}
	*model->output_neurons = ns_output;
}

void layer_add_current_neurons(NS_LAYER* layer, NS_NEURON* neuron)
{
	ns_array_append_no_duplicate(layer, neuron);
	if (!neuron->role & ROLE_OUTPUT)
		for (uint64_t i = 0; i < neuron->n_children; ++i)
			layer_add_current_neurons(layer, neuron->children[i]);
	else
		ns_array_append_no_duplicate(layer, neuron);
}

NS_LAYER* model_get_all_neurons(NS_MODEL* model)
{
	NS_LAYER* layer = malloc(sizeof(NS_LAYER));
	for (uint64_t i = 0; i < model->n_input_neurons; ++i)
		layer_add_current_neurons(layer, model->input_neurons[i]);
	return layer;
}

void delete_synapse(NS_SYNAPSE* synapse)
{
	if (!synapse)
		return;
	synapse->input_value = 0;
	// sets a free space in array for both parent and child neurons
	for (uint64_t i = 0; i < synapse->parent->n_children; ++i)
		if (synapse->parent->children[i] == synapse)
		{
			synapse->parent->children[i] = 0;
			synapse->parent->n_children--;
		}
	synapse->parent = 0;
	for (uint64_t i = 0; i < synapse->child->n_parents; ++i)
		if (synapse->child->parents[i] == synapse)
		{
			synapse->child->parents[i] = 0;
			synapse->child->n_parents--;
		}
	synapse->child = 0;
	synapse->weight = 0;
	free(synapse);
}

void delete_neuron(NS_NEURON* neuron)
{
	if (!neuron)
		return;
	for (uint64_t i = 0; i < neuron->n_parents; ++i)
		delete_synapse(neuron->parents[i]);
	for (uint64_t i = 0; i < neuron->n_children; ++i)
		delete_synapse(neuron->children[i]);
	neuron->bias = 0;
	neuron->children = 0;
	neuron->parents = 0;
	neuron->delta = 0;
	neuron->function = 0;
	neuron->id = 0;
	neuron->value = 0;
	neuron->n_parents = 0;
	neuron->n_children = 0;
	neuron->role = 0;
	free(neuron);

	NEURON_NUMBER--;
}

void delete_layer(NS_LAYER* layer)
{
	for (uint64_t i = 0; i < layer->size; ++i)
		delete_neuron(layer->elements[i]);
}

void delete_model(NS_MODEL* model)
{
	delete_layer(model_get_all_neurons(model));
}
