import numpy as np
import unittest
from model import Model
from database import Wheat, MNIST

class TestMNIST(unittest.TestCase):
	def test_dataset(self):
		db = MNIST()
		X_train, X_test, Y_train, Y_test = db.get_data()
		assert X_train.shape == (60000, 28*28, 1)
		assert Y_train.shape == (60000, 10, 1)

	def test_model(self):
		db = MNIST()
		X_train, X_test, Y_train, Y_test = db.get_data()

		NN = Model()

		model = NN.Input(128, input_shape=X_train.shape[1], activation='ReLu')		
		model = NN.Dense(128, 100, model, activation='ReLu')
		model = NN.Dense(100, 50, model, activation='ReLu')
		model = NN.Dense(50, 30, model, activation='ReLu')
		model = NN.Output(30, 10, model, activation='Softmax')
		print('hey', model[0][0].shape, model[1][0].shape, model[2][0].shape
			, model[2][0].shape, model[3][0].shape, model[4][0].shape)

		#train the model
		loss, accuracy = NN.Train(model, X_train, Y_train, 
			loss='MSE', opt='SGD', epochs=20, batch=256, categoric=True)

		self.assertEqual(len(loss), 20)
		self.assertEqual(len(accuracy), 20)
		self.assertFalse(np.any(np.array(accuracy) > 1))
		self.assertEqual(len(model), 5)
		self.assertEqual(model[0][2], 'ReLu')
		self.assertEqual(model[-1][2], 'Softmax')
		self.assertEqual(model[1][0].shape, (100, 128)) #weights of 1st hidden layer
		self.assertEqual(model[0][1].shape, (128, 1)) #bias shape of input
		self.assertEqual(model[-1][0].shape[0], 10) #number of outputs


class TestWheat(unittest.TestCase):	
	def test_dataset(self):
		db = Wheat()
		X_train, X_test, Y_train, Y_test = db.get_data(train=0.9)
		assert X_train.shape == (189, 7, 1) 
		assert Y_train.shape == (189, 1)

	def test_model(self):
		db = Wheat()
		X_train, X_test, Y_train, Y_test = db.get_data(train=0.9)

		NN = Model()

		model = NN.Input(10, input_shape=X_train.shape[1], activation='ReLu')
		model = NN.Dense(10, 7, model, activation='ReLu')
		model = NN.Dense(7, 5, model, activation='ReLu')
		model = NN.Output(5, 1, model, activation='Linear')

		#train the model
		loss, accuracy = NN.Train(model, X_train, Y_train, 
			loss='MSE', opt='SGD', epochs=10, batch=1, categoric=False)	


		self.assertEqual(len(loss), 10)
		self.assertEqual(len(accuracy), 10)
		self.assertFalse(np.any(np.array(accuracy) > 1))
		self.assertEqual(len(model), 4)
		self.assertEqual(model[0][2], 'ReLu')
		self.assertEqual(model[0][0].shape, (10, 7)) #weights shape of input
		self.assertEqual(model[0][1].shape, (10, 1)) #bias shape of input
		self.assertEqual(model[-1][0].shape[0], 1) #number of outputs


if __name__ == '__main__':
	unittest.main()		