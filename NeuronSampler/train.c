#include "train.h"

void train_model(NS_MODEL* model, NS_TARGET* target, uint64_t epoch, double precision)
{
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

void mass_train_model(NS_MODEL* model, NS_DATASET* dataset, uint64_t epoch, double learning_rate)
{
	for (int i = 0; i < dataset->size; ++i)
		train_model(model, dataset->elements[i], epoch, learning_rate);
}

void model_query(NS_MODEL* model, NS_TARGET* input)
{
	model_feed_values(model, input);
	int n_neurons = model->n_output_neurons;
	for (int _out = 0; _out < n_neurons; ++_out)
		neuron_forward(model->output_neurons[_out]);

}

void save_model_state(NS_MODEL* model, FILE* stream)
{
	if (!stream)
		return;
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
	if(ret)
		free(ret);
	end = clock();
	return end - start;
}

clock_t benchmark_training(void (*training)(NS_MODEL*, NS_TARGET*, uint64_t, double), NS_MODEL* nsm, NS_TARGET* nst, uint64_t epoch, double learning_rate)
{
	clock_t
		start = clock(),
		end;
	training(nsm, nst, epoch, learning_rate);
	end = clock();
	return end - start;
}

clock_t benchmark_model_creation(NS_MODEL*(*model_creation_function)(void))
{
	clock_t
		start = clock(),
		end;
	NS_MODEL* _model = model_creation_function();
	debug("Finished model creation execution\n");
	end = clock();
	delete_model(_model);
	debug("Deleted model \n");
	return end - start;
}
