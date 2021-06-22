import numpy as np 
import activation
import losses
import optimizers

class Model():
	def __init__(self):
		pass


	def forward(self, model, x):
		# W -- weight nxm matrix of for the next layer
		# x -- input vector of shape (m,1)
		# b -- bias vector of shape (n,1)
		# n -- number of output neurons (neurons of next layer)
		# m -- number of input neurons (neurons in the previous layer)
		# z[i] = W[i-1]x + b[i-1]
		# a[i] = f(z[i]) -- f is the activation funciton
		W = model[0]
		b = model[1]
		actv = model[2]
		act = getattr(activation, actv)
		res = np.dot(W,x) + b
		return [x, res, act(res)]


	def backward(self, A, actv, y, loss, output, next_model, all_loss):
		# we do backpropagation
		# Output layer:
		# dc_dw_o = (dc_da_o)*(da_dz_o)*(dz_dw_o)
		# dc_db_o = (dc_da_o)*(da_dz_o)(dz_db_o) -- (dz_db_o) == 1 because z = wx+b
		# Hidden/Input layer(s):
		# dc_dw = (dc_da)*(da_dz)*(dz_dw)
		# but dc_da = (dc_da_t1)*(da_t1_dz_t1)*(dz_t1_da), where t1 means one layer above
		# dc_db_ = (dc_da)*(da_dz)(dz_db) -- (dz_db) == 1 because z = wx+b

		W_t1 = next_model[0]
		a_t0 = A[0]
		z = A[1]
		a = A[2]
		d_loss_ = getattr(losses, 'd'+loss)
		d_act_ = getattr(activation, 'd'+actv)

		if output:
			all_loss = list()
			#dc_dw
			dc_da_o = d_loss_(a, y)
			da_dz_o = d_act_(z)
			dz_dw_o = a_t0.T #previous activation
			dc_dw_o = np.dot(dc_da_o*da_dz_o, dz_dw_o)

			#dc_db
			dc_db_o = dc_da_o*da_dz_o

			print('output')
			print(dc_da_o, da_dz_o, dc_dw_o, dc_db_o)
			return [dc_da_o, da_dz_o, dc_dw_o, dc_db_o]

		else:
			#dc_dw
			dc_da_t1 = all_loss[0]
			da_t1_dz_t1 = all_loss[1]
			dz_t1_da = W_t1
			dc_da = np.dot((dc_da_t1*da_t1_dz_t1).T, dz_t1_da).T
			da_dz = d_act_(z)
			dz_dw = a_t0.T
			dc_dw = np.dot(dc_da*da_dz, dz_dw)

			#dc_db
			dc_db = dc_da*da_dz

			print('other layers')
			print(dc_da, da_dz, dc_dw, dc_db)
			return [dc_da, da_dz, dc_dw, dc_db]


	def summary(self, model):
		print(f'| Total Number of Layers: {len(model)} |')
		for i in range(len(model)):
			inputs = model[i][0].shape[1]
			outputs = model[i][0].shape[0]

			print('| layer {} with {} inputs and {} outputs neurons |'.format(i+1, 
				inputs, outputs))

	def Input(self, neurons, input_shape, activation):
		#random weights and bias between -0.5 to 0.5
		weights = np.random.rand(neurons, input_shape) - 0.5
		bias = np.random.rand(neurons, 1) - 0.5
		return [[weights, bias, activation]]

	def Dense(self, pr_neurons, next_neurons, model, activation):
		weights = np.random.rand(next_neurons, pr_neurons) - 0.5
		bias = np.random.rand(next_neurons, 1) - 0.5
		model.append([weights, bias, activation])
		return model

	def Output(self, pr_neurons, next_neurons, model, activation):
		weights = np.random.rand(next_neurons, pr_neurons) - 0.5
		bias = np.random.rand(next_neurons, 1) - 0.5
		model.append([weights, bias, activation])
		return model

	def Train(self, model, x, y, loss, opt, epochs, batch, lr=0.04):
		l = []
		ac = []
		
		self.summary(model)
		#print('model summary')
		for i in range(epochs):
			avg_loss = 0
			acc = 0
			for k in np.arange(0, y.shape[0], batch): #updating only batch-size data
				A = list()
				
				#forward pass
				for j in range(len(model)):
					if (j == 0):
						A.append(self.forward(model[j], x[k]))
					else:
						A.append(self.forward(model[j], A[-1][2]))

				#backward pass
				#loss
				loss_ = getattr(losses, loss)
				#print(f'predicted results: {A[-1][2]}')
				#print(f'real results: {y[k]}')
				avg_loss += loss_(A[-1][2], y[k])
				
				#m = np.argmax(y[k])
				#if (np.argmax(A[-1][2]) == m):
				#	acc += 1
				#print(f'guess: {A[-1][2]}, true: {y[k]}, loss: {loss_(A[-1][2], y[k])}')

				if (A[-1][2][0] == y[k]):
					acc += 1


				all_loss = list()
				for j in range(len(model)):
					if j == 0:
						all_loss.append(self.backward(A[-1-j], model[-1-j][2], y[k], loss, True, model[-j], [0,0,0]))
					else:
						all_loss.append(self.backward(A[-1-j], model[-1-j][2], y[k], loss, False, model[-j], all_loss[-1]))
					
				opt_ = getattr(optimizers, opt)
				#print(f'updating weights: {all_loss[0][2]}')
				#if len( np.argwhere(all_loss[2][2] > 10) != 0):
				#	print('large numbers', np.argwhere(all_loss[2][2] > 10))
				model = opt_(model, all_loss, lr)
			acc /= len(np.arange(0, y.shape[0], batch))
			avg_loss /= len(np.arange(0, y.shape[0], batch))

			print(f'epoch: {i}, accuracy: {acc}, loss: {avg_loss}')
			l.append(avg_loss)
			ac.append(acc)

		return [l, ac]