import numpy as np

def MSE(x, y):
	return np.sum((x-y)**2)/len(x)

def dMSE(x,y):
	return 2*(x-y)

def Categorical_Cross_Entropy(x, y):
	return -x*np.log(y)

def dCategorical_Cross_Entropy(x, y):
	return x-y