import numpy as np

def SGD(model, gradients, lr):
	for i in range(len(model)):
		model[i][0] -= lr*gradients[-1-i][2] #updating weights
		model[i][1] -= lr*gradients[-1-i][3] #updating biases

	return model

