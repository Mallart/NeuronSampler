#include "neuron.h"
#define DEBUG

#ifdef DEBUG
void tests()
{
	NS_NEURON* input[]  = { create_neuron(), create_neuron() };
	NS_NEURON* layer1[] = { create_neuron(), create_neuron(), create_neuron() };
	NS_NEURON* layer2[] = { create_neuron(), create_neuron(), create_neuron() };
	NS_NEURON* output[] = { create_neuron() };
	BIND_CONST_LAYERS(input, layer1);
	BIND_CONST_LAYERS(layer1, layer2);
	BIND_CONST_LAYERS(layer2, output);
	const uint64_t n = sizeof(input) / sizeof(NS_NEURON*);
	NS_MODEL* model = create_model(input, n);
	return;
}
#endif

int main()
{
#ifdef DEBUG
	tests();
#endif
	return 0;
}