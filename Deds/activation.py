import numpy as np

def ReLu(x):
	return np.maximum(0,x)
	#return x if x > 0 else 0

def dReLu(x):
	data = np.copy(x)
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			if(x[i] > 0):
				data[i][j] = 1
			else:
				data[i][j] = 0
	return data

def Sigmoid(x):
	return 1/(1 + np.exp(-x))

def dSigmoid(x):
	return sigmoid(x)*(1-sigmoid(x))

def Softmax(x):
	max = np.max(x)

	if max >= 100:
		res = np.zeros((len(x),1))
		res[np.argmax(x)] = 1
		return res

	return np.exp(x-max)/sum(np.exp(x-max))

#res = np.exp(x)/(np.sum(np.exp(x)))
#res = np.clip(res, -88.72, 88.72)
#return res

def Tanh(x):
	(np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
