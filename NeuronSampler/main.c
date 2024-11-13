#include "train.h"
#define DEBUG

#ifdef DEBUG

/* TODO

- Serialization / deserialization
- Finishing training (implementing backpropagation compared to an ideal model NS_TARGET)
- Mass training using several targets (dataset)
- Multithreading
- GPU support

FIX

- Problem with freeing neuron with id 14 in first instanciation, check why this can happen
- layer_add_current_neurons has a hard time retrieving all the neurons without error

*/

NS_MODEL* example_model()
{
	// first, we create layers
	NS_NEURON* input[]  = { create_neuron(), create_neuron() };
	NS_NEURON* layer1[] = { create_neuron(), create_neuron(), create_neuron() };
	NS_NEURON* layer2[] = { create_neuron(), create_neuron(), create_neuron() };
	NS_NEURON* layer3[] = { create_neuron(), create_neuron(), create_neuron() };
	NS_NEURON* layer4[] = { create_neuron(), create_neuron(), create_neuron() };
	NS_NEURON* output[] = { create_neuron() };

	// then, we choose an activation function for each neuron / layer

	// this one is automatically set when we create the model
	layer_set_function(raw, input, CONST_LAYER_SIZE(input));
	layer_set_function(tanh, layer1, CONST_LAYER_SIZE(layer1));
	layer_set_function(tanh, layer2, CONST_LAYER_SIZE(layer2));
	layer_set_function(tanh, layer3, CONST_LAYER_SIZE(layer3));
	layer_set_function(tanh, layer4, CONST_LAYER_SIZE(layer4));
	layer_set_function(raw, output, CONST_LAYER_SIZE(output));
	
	// we have to bind layers together
	BIND_CONST_LAYERS(input, layer1);
	BIND_CONST_LAYERS(layer1, layer2);
	BIND_CONST_LAYERS(layer2, layer3);
	BIND_CONST_LAYERS(layer3, layer4);
	BIND_CONST_LAYERS(layer4, output);

	// then we create the model using the input layer
	return create_model(input, CONST_LAYER_SIZE(input), output, CONST_LAYER_SIZE(output));
}

NS_TARGET example_target()
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
	NS_TARGET target =
	{
		.n_inputs = CONST_ARRAY_SIZE(double, t_inputs),
		.n_outputs = CONST_ARRAY_SIZE(double, t_output),
		.inputs = t_inputs,
		.outputs = t_output
	};
	return target;
}


void serialization_test()
{
	NS_TARGET target = example_target();
	NS_MODEL* model = example_model();
	model_feed_values(model, &target);
	//train_model(model, &target, 10000, .003);
	char* buffer = serialize_model(model);
	NS_MODEL* copy = deserialize_model(buffer);
}

void test()
{
	// since I freed the model from memory, some uncanny crashes occur... funny haha
	clock_t model_creation = benchmark_model_creation(example_model);
	printf("Model creation time (ms): %i\n", model_creation);
	clock_t serialization_time = benchmark(serialization_test);
	printf("Model serialization time (ms): %i\n", serialization_time);
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
	// it appears that the model's output neuron is freed near here for an unkown reason. 
	// That's why the program crashes.
	clock_t training = benchmark_training(train_model, test_model, &target, 100000, .001);
	NS_NEURON* _output = test_model->output_neurons[0];
	printf("Model training time (ms): %i\n\nFirst neuron output value: %f \nbias: %f\nweight: %f\n", 
		training, 
		_output->value,
		_output->bias,
		_output->parents[0]->weight
	);

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