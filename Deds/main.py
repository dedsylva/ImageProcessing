#from keras.datasets import mnist
#from keras.utils import to_categorical
import numpy as np
from model import Model

#load data
#(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#need that channel dimension, normalized float32 tensor
#X_train = train_images.reshape((60000, 28*28, 1)).astype('float32')/255 
#Y_train =  to_categorical(train_labels).reshape((60000, 10, 1))
#X_test = test_images.reshape((10000, 28*28)).astype('float32')/255 
#Y_test =  to_categorical(test_labels).reshape((10000, 10, 1))

data_x = np.zeros((500, 2))
y = []
count = 0
for i in np.linspace(1.0, 1000.0, num = 500):
	data_x[count] = [i**2, i]
	y.append(i**2 + 3*i + 9)
	count += 1

#data_x = np.array(x).reshape(len(x), 2*len(x))
data_y = np.array(y).reshape(len(y), 1)

X_train = data_x[100:]
Y_train = data_y[100:]
X_test = data_x[:100]
Y_test = data_y[:100]

#Y = x**2 + 3*x + 9

NN = Model()
# model is a list where:
# len(model) = number of layers (counting the input and output)
# model[i][0] is the Weights matrix of layer i
# model[i][1] is the bias vector of layer i
# model[i][2] is the activation of the layer i <<-- limitation: currently we can only have the same activation for the entire layer
model = NN.Input(5, input_shape=X_train.shape[0], activation='ReLu')
#model = NN.Dense(5, 3, model, activation='ReLu') #input neurons: 10 output neurons: 5
#model = NN.Dense(5, 7, model, activation='ReLu')
model = NN.Output(5, 1, model, activation='Softmax')

#train the model
loss, accuracy = NN.Train(model, X_train, Y_train, 
	loss='MSE', opt='SGD', epochs=5, batch=1)