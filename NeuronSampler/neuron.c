#include "neuron.h"

void* array_append(void** array, uint64_t array_size, void* element)
{
	uint64_t i = 0;
	for (; i < array_size; ++i)
		if (!array[i])
			break;
	// full array, have to realloc
	if (i == array_size - 1)
	{
		array = realloc(array, array_size + sizeof(element));
		array[array_size] = element;
		return array;
	}
	// free space detected, put a pointer in it
	array[i] = element;
	return 0;
}

void* array_append_no_duplicate(void** array, uint64_t array_size, void* element)
{
	if (!array_exists(array, array_size, element))
		return array_append(array, array_size, element);
	return array;
}

void array_remove(void** array, uint64_t array_size, void* element)
{
	for (uint64_t i = 0; i < array_size; ++i)
		if (array[i] == element)
			array[i] = 0;
}

bool array_exists(void** array, uint64_t array_size, void* element)
{
	for (uint64_t i = 0; i < array_size; ++i)
		if (array[i] == element)
			return true;
	return false;
}

NS_SYNAPSE* create_synapse(NS_NEURON* parent, NS_NEURON* child)
{
	NS_SYNAPSE* synapse = malloc(sizeof(NS_SYNAPSE));
	if (!synapse)
		return 0;
	synapse->parent = parent;
	synapse->child = child;
	
	if (!array_exists(parent->children, parent->n_children, child))
	{
		parent->children = array_append(parent->children, parent->n_children, synapse);
		parent->n_children++;
	}
	
	if (!array_exists(child->parents, child->n_parents, parent))
	{
		child->parents = array_append(child->parents, child->n_parents, synapse);
		parent->n_parents++;
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
	NS_NEURON* neuron = malloc(sizeof(NS_NEURON));
	if (!neuron)
		return 0;
	neuron->parents = malloc(sizeof(NS_SYNAPSE*));
	neuron->children = malloc(sizeof(NS_SYNAPSE*));
	neuron->n_parents = 0;
	neuron->n_children = 0;
	neuron->function = 0;
	return neuron;
}

void init_neuron(NS_NEURON* neuron)
{
	if(neuron->parents)
		free(neuron->parents);
	if(neuron->children)
		free(neuron->children);
	neuron->parents = malloc(sizeof(NS_SYNAPSE*));
	neuron->children = malloc(sizeof(NS_SYNAPSE*));
	neuron->n_parents = 0;
	neuron->n_children = 0;
	neuron->function = 0;
}