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
		array_append(array, array_size, element);
}

bool array_exists(void** array, uint64_t array_size, void* element)
{
	for (uint64_t i = 0; i < array_size; ++i)
		if (array[i] == element)
			return true;
	return false;
}

void bind_child(NS_NEURON* parent, NS_NEURON* child)
{
	if (!array_exists(parent->children, parent->n_children, child))
	{
		parent->children = array_append(parent->children, parent->n_children, child);
		parent->n_children++;
	}
	if (!array_exists(child->parents, child->n_parents, parent))
	{
		child->parents = array_append(child->parents, child->n_parents, parent);
		child->n_parents++;
	}
}
