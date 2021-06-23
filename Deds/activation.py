import numpy as np

def ReLu(x):
	return np.maximum(0,x)

def dReLu(x):
	data = np.copy(x)
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			if(x[i][j] > 0):
				data[i][j] = 1
			else:
				data[i][j] = 0
	return data

def Sigmoid(x):
	return 1/(1 + np.exp(-x))

def dSigmoid(x):
	return sigmoid(x)*(1-sigmoid(x))

def Linear(x):
	return x

def dLinear(x):
	x.fill(1)
	return x

def Softmax(x):
	max_ = np.max(x)

	if max_ >= 100:
		res = np.zeros((len(x),1))
		res[np.argmax(x)] = 1
		return res

	return np.exp(x-max_)/sum(np.exp(x-max_))

def dSoftmax(x):
	return x*(1-x)

def Tanh(x):
	(np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
