import numpy as np

def SGD(model, gradient, lr):
	for i in range(len(model)):
		model[0] -= lr*gradient[0] #updating weights
		model[1] -= lr*gradient[1] #updating biases

	return model

