#include "train.h"
#define DEBUG

#ifdef DEBUG

/* TODO

- Serialization / deserialization
- Finishing training (implementing backpropagation compared to an ideal model NS_TARGET)
- Multithreading
- GPU support

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

void test()
{
	clock_t model_creation = benchmark(example_model);
	printf("Model creation time (ms): %i\n", model_creation);
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
	printf("Model training time (ms): %i\n\nFirst neuron output value: %f \nbias: %f\nweight: %f", 
		training, 
		test_model->output_neurons[0]->value,
		test_model->output_neurons[0]->bias,
		test_model->output_neurons[0]->parents[0]->weight
	);
}
#endif

int main()
{
#ifdef DEBUG
	test();
#endif
	return 0;
}