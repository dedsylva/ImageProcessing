import numpy as np 
from keras.datasets import mnist
from keras.utils import to_categorical

def relu(x):
	return np.maximum(0,x)

def softmax(x):
	max_ = np.max(x)
	return np.exp(x-max_)/np.sum(np.exp(x-max_))

def dReLu(x):
	data = np.copy(x)
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			if(x[i][j] > 0):
				data[i][j] = 1
			else:
				data[i][j] = 0
	return data

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#need that channel dimension, normalized float32 tensor
X_train = train_images.reshape((60000, 28*28, 1)).astype('float32')/255 
Y_train =  to_categorical(train_labels).reshape((60000, 10, 1))
X_test = test_images.reshape((10000, 28*28, 1)).astype('float32')/255 
Y_test =  to_categorical(test_labels).reshape((10000, 10, 1))

x = X_train
y = Y_train

# layer 1
weights_1 = np.zeros((128, 28*28), dtype='float32')
b_1 = np.zeros((128, 1), dtype='float32')
weights_1 = np.random.rand(128, 28*28) - 0.5
b_1 = np.random.rand(128, 1) - 0.5
		
weights_2 = np.zeros((10, 128), dtype='float32')
b_2 = np.zeros((10, 1), dtype='float32')

weights_2 = np.random.rand(10, 128) - 0.5
b_2 = np.random.rand(10, 1) - 0.5

epochs = 200 
BS = 32
lr = 0.0001

losses = []
accuracy =[]
for i in range(epochs):

	acc = 0
	loss = 0
	for k in np.arange(0, y.shape[0], BS):
		#forward
		z_1 = np.dot(weights_1, x[k])+ b_1
		a_1 = relu(z_1) 
		z_2 = np.dot(weights_2, a_1) + b_2
		a_2 = softmax(z_2)

		loss += (a_2 - y[k]).mean(axis=1)
		if (np.argmax(a_2) == np.argmax(y[k])):
			acc += 1

		#backward
		#output layer
		dc_da_o = 2*(a_2 - y[k])
		da_dz_o = a_2*(1-a_2)
		dz_dw_o = a_1.T
		dw2 = np.dot(dc_da_o*da_dz_o,dz_dw_o)
		db2 = dc_da_o*da_dz_o

		#other layers
		dc_da_t1 = dc_da_o
		da_t1_dz_t1 = da_dz_o
		dz_t1_da = weights_2
		dc_da = np.dot((dc_da_t1*da_t1_dz_t1).T, dz_t1_da).T
		da_dz = dReLu(a_1)
		dz_dw = x[k].T
		dw1 = np.dot(dc_da*da_dz, dz_dw)
		db1 = dc_da*da_dz

		#sgd
		weights_1 -= lr*dw1
		b_1 -= lr*db1
				
		weights_2 -= lr*dw2
		b_2 -= lr*db2

		#print(f'difference on weight: {dw2}\nmax number: {np.max(dw2, axis=1)}')
		#print(f'difference on weight: {dw2}\ndifference on bias: {db2}')
		#print(f'predic: {a_2}')

	acc /= len(np.arange(0, y.shape[0], BS))
	loss /= len(np.arange(0, y.shape[0], BS))
	accuracy.append(acc)
	losses.append(loss)
	print(f'epoch: {i}, accuracy: {acc}')
	#print(f'epoch: {i}, accuracy: {acc}, loss: {loss}')


print('Evaluating')
precision = 0
for k in range(len(Y_test)):
	z_1 = np.dot(weights_1, X_test[k])+ b_1
	a_1 = relu(z_1) 
	z_2 = np.dot(weights_2, a_1) + b_2
	a_2 = softmax(z_2)

	if (np.argmax(a_2) == np.argmax(Y_test[k])):
		precision += 1

print(f'##### Network got {precision/len(Y_test)} right')


import matplotlib.pyplot as plt
plt.plot(range(epochs), accuracy)
plt.title('accuracy during training')
plt.show()