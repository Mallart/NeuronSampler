#pragma once
#include "libs.h";

typedef struct
{
	uint64_t n_parents;
	uint64_t n_children;
	void* function;
	struct NS_NEURON* parents;
	struct NS_NEURON* children;
} NS_NEURON;

// appends an element on an array of pointers and returns the new array
// WARNING: can destroy the old array if there's not enough free space to add an element
void* array_append(void** array, uint64_t array_size, void* element);
void* array_append_no_duplicate(void** array, uint64_t array_size, void* element);
// returns true if the specified element exist in array
bool array_exists(void** array, uint64_t array_size, void* element);

// creates a new neuron
NS_NEURON* create_neuron();
// initiates a blank neuron or reset an existing one.
void init_neuron(NS_NEURON* neuron);
// bind a child to his new parent if they're not bound already.
void bind_child(NS_NEURON* parent, NS_NEURON* child);