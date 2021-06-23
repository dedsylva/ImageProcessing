import sys
import numpy as np
from model import Model
import pandas as pd
import math
from database import Wheat, MNIST

def main(argv):
	data = [a.split('=') for a in argv]

	if data[0][0] == 'model':
		if data[0][1] == 'MNIST':
			db = MNIST()
			X_train, X_test, Y_train, Y_test = db.get_data()

			NN = Model()

			model = NN.Input(128, input_shape=X_train.shape[1], activation='ReLu')
			model = NN.Dense(128, 100, model, activation='ReLu')
			model = NN.Dense(100, 50, model, activation='ReLu')
			model = NN.Dense(50, 30, model, activation='ReLu')
			model = NN.Output(30, 10, model, activation='Softmax')

			#train the model
			loss, accuracy = NN.Train(model, X_train, Y_train, 
				loss='MSE', opt='SGD', epochs=20, batch=256, categoric=True)

		elif data[0][1] == 'Wheat':
			db = Wheat()
			train = float(data[1][1]) if len(data) > 1 else 0.9
			X_train, X_test, Y_train, Y_test = db.get_data(train=train)
			
			NN = Model()

			model = NN.Input(10, input_shape=X_train.shape[1], activation='ReLu')
			model = NN.Dense(10, 7, model, activation='ReLu')
			model = NN.Dense(7, 5, model, activation='ReLu')
			model = NN.Output(5, 1, model, activation='Linear')

			#train the model
			loss, accuracy = NN.Train(model, X_train, Y_train, 
				loss='MSE', opt='SGD', epochs=10, batch=1, categoric=False)				
		else:
			raise Exception ('The Model you entered are not available')
	else:
		raise ValueError ('The argument \'{}\' is invalid!'.format(data[0][0]))

if __name__ == '__main__':
	main(sys.argv[1:])