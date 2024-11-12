#include "train.h"

void train_model(NS_MODEL* model, NS_TARGET* target, uint64_t epoch, double precision)
{
	//NS_VALUE** target_values = target->inputs;
	for(uint64_t _e = 0; _e < epoch; ++_e)
		for (uint64_t i = 0; i < model->n_output_neurons; ++i)
		{
			neuron_forward(model->output_neurons[i]);
			if(target)
				neuron_backwards(model->output_neurons[i], target->outputs[i], precision);
			else
				neuron_backwards(model->output_neurons[i], REPLACE_THIS_VALUE_WITH_WORKING_ONE, precision);
		}
}

void save_model_state(NS_MODEL* model, FILE* stream)
{
	if (!stream)
		return 0;
}

NS_MODEL* read_model_state(FILE* stream)
{
	if (!stream)
		return 0;
}

clock_t benchmark(void* (*f)(void))
{
	clock_t
		start = clock(),
		end;
	void* ret = f();
	free(ret);
	end = clock();
	return end - start;
}

clock_t benchmark_training(void (*training)(NS_MODEL*, NS_TARGET*, uint64_t), NS_MODEL* nsm, NS_TARGET* nst, uint64_t epoch)
{
	clock_t
		start = clock(),
		end;
	training(nsm, nst, epoch);
	end = clock();
	return end - start;
}

clock_t benchmark_model_creation(NS_MODEL*(*model_creation_function)(NS_MODEL*, NS_TARGET*, uint64_t), NS_MODEL* nsm, NS_TARGET* nst, uint64_t epoch)
{
	clock_t
		start = clock(),
		end;
	NS_MODEL* _model = model_creation_function(nsm, nst, epoch);
	end = clock();
	delete_model(_model);
	return end - start;
}
