import numpy as np
from model import Model
import pandas as pd
import math

'''
#load data

from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#need that channel dimension, normalized float32 tensor
X_train = train_images.reshape((60000, 28*28, 1)).astype('float32')/255 
Y_train =  to_categorical(train_labels).reshape((60000, 10, 1))
X_test = test_images.reshape((10000, 28*28)).astype('float32')/255 
Y_test =  to_categorical(test_labels).reshape((10000, 10, 1))
#Y_train = Y_train[:,:2,:]

'''
#loads data
df = pd.read_csv('seeds_dataset.csv')
m = df.shape[0]
n = df.shape[1]
data = np.zeros((m,n))

#to numpy
for i in range(len(df)):
	data[i] = df.loc[i].to_numpy()

#replacing possible 0,nans with mean of each feature
for i in range(data.shape[1]):
	a = data[:,i]
	mean_i = a.mean()
	a = np.nan_to_num(a)
	a[a==0] = mean_i



np.random.shuffle(data)
train = math.ceil(0.9*m)

X_train = data[:train,:-1].reshape(train, n-1, 1)
X_test = data[train:,:-1].reshape(m-train,n-1, 1)
Y_train = data[:train,-1].reshape(train,1)
Y_test = data[train:,-1].reshape(m-train,1)

NN = Model()
# model is a list where:
# len(model) = number of layers (counting the input and output)
# model[i][0] is the Weights matrix of layer i
# model[i][1] is the bias vector of layer i
# model[i][2] is the activation of the layer i <<-- limitation: currently we can only have the same activation for the entire layer
model = NN.Input(10, input_shape=X_train.shape[1], activation='ReLu')
#model = NN.Dense(5, 3, model, activation='ReLu') #input neurons: 10 output neurons: 5
#model = NN.Dense(100, 50, model, activation='ReLu')
model = NN.Output(10, 1, model, activation='ReLu')

#train the model
loss, accuracy = NN.Train(model, X_train, Y_train, 
	loss='MSE', opt='SGD', epochs=5, batch=1)