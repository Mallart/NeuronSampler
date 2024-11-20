# NeuronSampler
*An efficient and simple Software Development Kit to work with AI in C*
## What is NeuronSampler?
NeuronSampler is a library, or a software development kit intended to help developers integrate AI within their projects or just try and test things with AI.
## How does it work and how do I use it?
Simply: it's just basic AI.<br>
You can setup input nodes (or neurons), store them in layers, repeat the same process for hidden and output neurons, use functions to bind them all, and create **Targets**.<br>
A **Target** is basically a small dataset, holding both input and output data.<br>
You can use a **Target** to train your neural network, saying that using some given input data, you expect that output.<br>
Before finishing your model creation, you shall bind functions to your neurons (you can also bind them in bulk, using layers). You can now create your model !<br>
You then proceed to train your model until you're satisfied with the results it gives you.<br>
It's highly recommended to use several **Targets** to train your model, as it will only give one single answer to every question otherwise.<br>
## How do I query my model ?
You can query your model using appropriate functions and a **Target** with undefined expected output. The model will use forward pass technique to reflect with all necessary neurons and then give an appropriate result without changing any weight or bias in the model, accuracy depending on the amount of training and variety of data previously fed to the model.
## What are the applications for this ?
You can use this project as a library in any other project. Then, you can build software based on AI or just using AI without really having to understand the underlying mechanics of artificial intelligence.
## How does it learn ?
Without making a whole course on AI, given a dataset of variable size that fits the neural network and expecting answers from the model each time you make it learn with that dataset, it will compute the error size ('delta') and adjust the output neuron's bias and its parent synapse's weight in order to be closer to the expected answer. It will then 'backpropagate' these changes so each parent neuron will adjust their parameters to comply with the expected behavior of the model.
## What's the future of this project ?
This project is not finished yet at all.<br>
I'm planning on:
- Writing a comprehensive documentation on how to use this tool.
- Adding a functionality to save neural networks models and their states (biases, weights, functions...) and load them from disk.
- Adding a support for multithreading in order to train models quicker and select best among their generations of models.
- Adding a support for other types of neural networks (Convolutional Neural Networks, Recurrent Neural Networks, Transformers...).
- Adding a basic support for GPU-based computing (which can, in some cases, speed up model training).
- Creating a separate project which is an IDE for neural networks using this project as a core library.
- Make this README file prettier.