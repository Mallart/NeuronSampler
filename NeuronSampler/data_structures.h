#pragma once
#include "libs.h"

//typedef thrd_t thread;

typedef struct
{
	uint64_t size;
	void** elements;
} NS_ARRAY;

/*
	Context dependant variables
*/

typedef NS_ARRAY NS_NEURON_ARRAY;
typedef NS_ARRAY NS_SYNAPSE_ARRAY;
typedef NS_ARRAY NS_MODEL_ARRAY;

// appends an element on an array of pointers and returns the new array
// WARNING: can destroy the old array if there's not enough free space to add an element
void* array_append(void** array, uint64_t array_size, void* element);
void* array_append_no_duplicate(void** array, uint64_t array_size, void* element);
// replaces an element with 0.
void array_remove(void** array, uint64_t array_size, void* element);
// returns true if the specified element exist in array
bool array_exists(void** array, uint64_t array_size, void* element);

NS_ARRAY* ns_array_create();
// appends an element on an array of pointers and returns the new array
// WARNING: can destroy the old array if there's not enough free space to add an element
NS_ARRAY* ns_array_append(NS_ARRAY* array, void* element);
NS_ARRAY* ns_array_append_no_duplicate(NS_ARRAY* array, void* element);
// replaces an element with 0.
void ns_array_remove(NS_ARRAY* array, void* element);
// returns true if the specified element exist in array
bool ns_array_exists(NS_ARRAY* array, void* element);