cmake_minimum_required(VERSION 3.29)
project(NeuronSampler C)

set(CMAKE_C_STANDARD 11)

include_directories(NeuronSampler)

add_executable(NeuronSampler
    NeuronSampler/data_structures.c
    NeuronSampler/data_structures.h
    NeuronSampler/libs.h
    NeuronSampler/main.c
    NeuronSampler/neuron.c
    NeuronSampler/neuron.h
    NeuronSampler/neuron_functions.c
    NeuronSampler/neuron_functions.h
    NeuronSampler/train.c
    NeuronSampler/train.h
        NeuronSampler/debug.h)
