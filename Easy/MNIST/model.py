### Conv2D layers at keras (aka the filter construction layer)
# -> layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1))
# means 32 filters of 3x3 that will go through the images

### Max Pooling at keras (aka the downsizing layer)
# -> layers.MaxPooling2D((2,2))
# 2x2 groups that returns only the pixel which has maximum value of the filter defined in the layer above
# padding is set to same for deault, i.e, “pad in such a way as to have an output with the same width# - François Chollet - Deep Learning with python-Manning (2018)
# the results are half the size 

from keras import layers, models

class Model():
	def __init__(self, input_shape, optimizer, loss, metrics):
		self.input_shape = input_shape
		self.optimizer = optimizer
		self.loss = loss
		self.metrics = metrics

	## Salvando a rede ##
	def save_model(self, model, path):
		tf.keras.models.save_model(
		model,
		path,
		overwrite=True,
		include_optimizer=True,
		save_format='h5'
	)



	def build_model(self, print_=True):
		#CNN part
		model = models.Sequential()
		model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=self.input_shape))
		model.add(layers.MaxPooling2D((2,2)))
		model.add(layers.Conv2D(64, (3,3), activation='relu'))
		model.add(layers.MaxPooling2D((2,2)))
		model.add(layers.Conv2D(64, (3,3), activation='relu'))
		model.add(layers.MaxPooling2D((2,2)))

		#Classification part
		model.add(layers.Flatten())
		model.add(layers.Dense(50, activation='relu'))
		model.add(layers.Dense(10, activation='softmax'))

		model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
		
		if(print_):
			print(f'Model Summary')
			model.summary()

		return model

	def train(self, model, X_train, Y_train, epochs, batch):
		history = model.fit(
			x=X_train, y=Y_train, 
			epochs=epochs, batch_size=batch
		)
		return history