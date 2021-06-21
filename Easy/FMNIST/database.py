from keras.datasets import fashion_mnist
from keras.utils import to_categorical

class Database():
	def __init__(self):
		pass

	def get_data(self):
		#load data
		(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

		#need that channel dimension, normalized float32 tensor
		X_train = train_images.reshape((60000, 28, 28, 1)).astype('float32')/255 
		Y_train =  to_categorical(train_labels)
		X_test = test_images.reshape((10000, 28, 28, 1)).astype('float32')/255 
		Y_test =  to_categorical(test_labels)


		return X_train, Y_train, X_test, Y_test