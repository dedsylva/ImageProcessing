#import pandas as pd
import numpy as np
from model import Model

"""
data = pd.read_csv('').to_numpy()

np.random.shuffle(data)

d_test = data[:1000].T
X_test = d_test[1:]
Y_test = d_test[0]

d_train = data[1000:].T
X_train = d_train[1:]
Y_train = d_tran[0]
"""

X_train = np.random.rand(100, 28*28, 1,).astype('float32')
Y_train = np.zeros((100, 10, 1), dtype='float32')
for i in range(Y_train.shape[0]):
	Y_train[i][np.random.randint(10)] = 1./255

NN = Model()
# model is a list where:
# len(model) = number of layers (counting the input and output)
# model[i][0] is the Weights matrix of layer i
# model[i][1] is the bias vector of layer i
# model[i][2] is the activation of the layer i <<-- limitation: currently we can only have the same activation for the entire layer
model = NN.Input(10, input_shape=28*28, activation='ReLu')
model = NN.Dense(10, 5, model, activation='ReLu') #input neurons: 10 output neurons: 5
model = NN.Dense(5, 7, model, activation='ReLu')
model = NN.Output(7, 10, model, activation='Softmax')

#train the model
loss, accuracy = NN.Train(model, X_train, Y_train, 
	loss='MSE', opt='SGD', epochs=10, batch=6)