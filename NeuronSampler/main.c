#include "neuron.h"
#define DEBUG

#ifdef DEBUG
void example_model()
{
	// first, we create layers
	NS_NEURON* input[]  = { create_neuron(), create_neuron() };
	NS_NEURON* layer1[] = { create_neuron(), create_neuron(), create_neuron() };
	NS_NEURON* layer2[] = { create_neuron(), create_neuron(), create_neuron() };
	NS_NEURON* output[] = { create_neuron() };

	// then, we choose an activation function for each neuron / layer
	layer_set_function(raw, input, CONST_LAYER_SIZE(input));
	layer_set_function(relu, layer1, CONST_LAYER_SIZE(layer1));
	layer_set_function(sigmoid, layer2, CONST_LAYER_SIZE(layer2));
	layer_set_function(sigmoid, output, CONST_LAYER_SIZE(output));
	
	// we have to bind layers together
	BIND_CONST_LAYERS(input, layer1);
	BIND_CONST_LAYERS(layer1, layer2);
	BIND_CONST_LAYERS(layer2, output);

	// then we create the model using the input layer
	NS_MODEL* model = create_model(input, CONST_LAYER_SIZE(output));
	return;
}
#endif

int main()
{
#ifdef DEBUG
	example_model();
#endif
	return 0;
}