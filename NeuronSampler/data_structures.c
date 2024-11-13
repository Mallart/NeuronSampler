#include "data_structures.h"

void* array_append(void** array, uint64_t array_size, void* element)
{
	if (!array)
	{
		array = malloc(array_size * sizeof(element));
		*array = element;
		return array;
	}
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
	return array;
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
	if (!array)
		return false;
	for (uint64_t i = 0; i < array_size; ++i)
		if (array[i] && array[i] == element)
			return true;
	return false;
}

void** array_create_from_ns_array(NS_ARRAY* array)
{
	void** arr = calloc(array->size, sizeof(array->elements));
	for (uint64_t i = 0; i < array->size; ++i)
		arr[i] = array->elements[i];
	return arr;
}

NS_ARRAY* ns_array_create()
{
	return calloc(1, sizeof(NS_ARRAY));
}

NS_ARRAY* ns_array_append(NS_ARRAY* array, void* element)
{
	if (!array->elements)
	{
		array->elements = malloc(array->size * sizeof(element));
		*array->elements = element;
		return array;
	}
	uint64_t i = 0;
	for (; i < array->size; ++i)
		if (!array->elements[i])
			break;
	// full array, have to realloc
	if (i == array->size - 1)
	{
		array->elements = realloc(array->elements, array->size + sizeof(element));
		array->elements[array->size] = element;
		return array;
	}
	// free space detected, put a pointer in it
	array->elements[i] = element;
	array->size++;
	return array;
}

NS_ARRAY* ns_array_append_no_duplicate(NS_ARRAY* array, void* element)
{
	if (!array_exists(array->elements, array->size, element))
	{
		array = ns_array_append(array, element);
		return array;
	}
	return array;
}

NS_ARRAY* ns_array_create_from_buffer(void** array, uint64_t size)
{
	NS_ARRAY* ns_array = ns_array_create();
	for (uint64_t i = 0; i < size; ++i)
		ns_array_append(ns_array, array[i]);
	return ns_array;
}

void ns_array_remove(NS_ARRAY* array, void* element)
{
	for (uint64_t i = 0; i < array->size; ++i)
		if (array->elements[i] == element)
			array->elements[i] = 0;
}

bool ns_array_exists(NS_ARRAY* array, void* element)
{
	if (!array->elements)
		return false;
	for (uint64_t i = 0; i < array->size; ++i)
		if ((void*)(array->elements + i) && array->elements[i] == element)
			return true;
	return false;
}
