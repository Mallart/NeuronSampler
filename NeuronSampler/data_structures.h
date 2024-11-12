#pragma once
#include "libs.h"

// when everything is still not set up and I needed a placeholder value.
#define REPLACE_THIS_VALUE_WITH_WORKING_ONE 0
#define CONST_ARRAY_SIZE(type, array) sizeof(array) / sizeof(type)
#define NS_EPSILON .0000000001
//typedef thrd_t thread;

typedef uint32_t NS_FLAG;

enum NS_NEURON_FLAGS
{
	// should never be set in a working neural network
	ERROR,
	ROLE_INPUT	= 0b0001,
	ROLE_OUTPUT = 0b0010,
	ROLE_HIDDEN	= 0b0100,
	// if its value has been computed at least once
	LIT_STATE	= 0b1000
};

typedef struct NS_ARRAY
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

typedef double* NS_VALUES_ARRAY;
typedef NS_ARRAY NS_TARGETS_ARRAY;

typedef NS_NEURON_ARRAY NS_LAYER;
// appends an element on an array of pointers and returns the new array
// WARNING: can destroy the old array if there's not enough free space to add an element
void* array_append(void** array, uint64_t array_size, void* element);
void* array_append_no_duplicate(void** array, uint64_t array_size, void* element);
// replaces an element with 0.
void array_remove(void** array, uint64_t array_size, void* element);
// returns true if the specified element exist in array
bool array_exists(void** array, uint64_t array_size, void* element);
void** array_create_from_ns_array(NS_ARRAY* array);

NS_ARRAY* ns_array_create();
// appends an element on an array of pointers and returns the new array
// WARNING: can destroy the old array if there's not enough free space to add an element
NS_ARRAY* ns_array_append(NS_ARRAY* array, void* element);
NS_ARRAY* ns_array_append_no_duplicate(NS_ARRAY* array, void* element);
NS_ARRAY* ns_array_create_from_buffer(void** array, uint64_t size);
// replaces an element with 0.
void ns_array_remove(NS_ARRAY* array, void* element);
// returns true if the specified element exist in array
bool ns_array_exists(NS_ARRAY* array, void* element);