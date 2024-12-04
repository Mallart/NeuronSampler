#include "train.h"
#define DEBUG

#ifdef DEBUG

/* TODO

- Serialization / deserialization
- Finishing training (implementing backpropagation compared to an ideal model NS_TARGET)
- Mass training using several targets (dataset)
- Multithreading
- GPU support
- GCC compilation

FIX

- Problem with freeing neuron with id 14 in first instanciation, check why this can happen
- layer_add_current_neurons has a hard time retrieving all the neurons without error

*/

NS_MODEL* example_model()
{
	// first, we create layers
	NS_NEURON* _input[]  = { create_neuron(), create_neuron() };
	NS_NEURON* _layer1[] = { create_neuron(), create_neuron(), create_neuron() };
	NS_NEURON* _layer2[] = { create_neuron(), create_neuron(), create_neuron() };
	NS_NEURON* _layer3[] = { create_neuron(), create_neuron(), create_neuron() };
	NS_NEURON* _layer4[] = { create_neuron(), create_neuron(), create_neuron() };
	NS_NEURON* _output[] = { create_neuron() };

	NS_LAYER* input	 = s_ns_array_create_from_buffer(_input, CONST_LAYER_SIZE(_input));
	NS_LAYER* layer1 = s_ns_array_create_from_buffer(_layer1, CONST_LAYER_SIZE(_layer1));
	NS_LAYER* layer2 = s_ns_array_create_from_buffer(_layer2, CONST_LAYER_SIZE(_layer2));
	NS_LAYER* layer3 = s_ns_array_create_from_buffer(_layer3, CONST_LAYER_SIZE(_layer3));
	NS_LAYER* layer4 = s_ns_array_create_from_buffer(_layer4, CONST_LAYER_SIZE(_layer4));
	NS_LAYER* output = s_ns_array_create_from_buffer(_output, CONST_LAYER_SIZE(_output));

	// then, we choose an activation function for each neuron / layer

	// this one (input) is automatically set to raw when we create the model
	ns_layer_set_function(&raw, input);
	ns_layer_set_function(&tanh, layer1);
	ns_layer_set_function(&tanh, layer2);
	ns_layer_set_function(&tanh, layer3);
	ns_layer_set_function(&tanh, layer4);
	ns_layer_set_function(&raw, output);
	
	// we have to bind layers together
	BIND_NS_LAYERS(input, layer1);
	BIND_NS_LAYERS(layer1, layer2);
	BIND_NS_LAYERS(layer2, layer3);
	BIND_NS_LAYERS(layer3, layer4);
	BIND_NS_LAYERS(layer4, output);

	// then we create the model using the input layer
	return create_model((NS_NEURON**)input->elements, NS_LAYER_SIZE(input), (NS_NEURON**)output->elements, NS_LAYER_SIZE(output));
}

NS_TARGET* example_target()
{
	double
		t_inputs[] =
	{
		4, 5
	},
		t_output[] =
	{
		9
	};
	NS_TARGET* target = malloc(sizeof(NS_TARGET));
	target->n_inputs = CONST_ARRAY_SIZE(double, t_inputs);
	target->n_outputs = CONST_ARRAY_SIZE(double, t_output);
	target->inputs = t_inputs;
	target->outputs = t_output;
	return target;
}


void* serialization_test()
{
	NS_MODEL* model = example_model();
	model_feed_values(model, example_target());
	train_model(model, example_target(), 10000, .003);
	char* buffer = serialize_model(model);
	NS_MODEL* copy = deserialize_model(buffer);
	return 0;
}

void test()
{
	MEM_TEST

	NS_MODEL* test_model = example_model();
	double
		t_inputs[] =
	{
		4, 5
	},
		t_output[] =
	{
		9
	};
	NS_TARGET target =
	{
		.n_inputs = CONST_ARRAY_SIZE(double, t_inputs),
		.n_outputs = CONST_ARRAY_SIZE(double, t_output),
		.inputs = t_inputs,
		.outputs = t_output
	};
	model_feed_values(test_model, &target);
	clock_t creation = benchmark_model_creation(example_model);
	clock_t training = benchmark_training(train_model, test_model, &target, 100000, .001);
	NS_NEURON* _output = test_model->output_neurons[0];
	printf("Model creation time (ms):%i\n", creation);
	printf("Model training time (ms): %i\n\nFirst neuron output value: %f \nbias: %f\nweight: %f\n",
		training,
		_output->value,
		_output->bias,
		((NS_SYNAPSE*)_output->parents->elements[0])->weight
	);

	// clock_t serialization_time = benchmark(serialization_test);
	// printf("Model serialization time (ms): %i\n", serialization_time);
	/*
	double _inputs[] = { 3, 5 };
	test_model->output_neurons[0] = _output;
	model_query(test_model, &(NS_TARGET){.inputs = _inputs, CONST_ARRAY_SIZE(double, _inputs)});
	printf("\nResult of 3 + 5: %f", test_model->output_neurons[0]->value);
	*/
}
#endif

int main()
{
#ifdef DEBUG
	test();
#endif
	return 0;
}