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
		return act(np.dot(W,x) + b)


	def backward(self, a, previous_x, y, loss, output, next_model, all_loss):
		
		#self, model, x, a, prv_model, previous_x, y, loss, output, next_model, all_loss
		# we do backpropagation
		# starting from the output layer, going through chain rule 
		# we have a = activation(z), where z = wx + b
		# to get the dC/dW (derivative of the cost function with respect to the weights)
		# dC/dW = (dC/da)*(da/dz)*(dz/dW)
		# but z = w*x+b --> dz/dW = x
		# then dC/dW = (dC/da)*(da/dz)*x
		# Analogously for the bias:
		# dC/db = (dC/da)*(da/dz)*(dz/db)
		# but z = wx+b --> dz/db = 1
		# then dC/db = (dC/da)*(da/dz)
		#W = model[0]
		#b = model[1]
		#actv = model[2]

		#loss_ = getattr(losses, loss)
		dloss_ = getattr(losses, 'd'+loss)
		d_act = getattr(activation, 'dReLu')
		
		if output:
			dC_dW = np.dot(dloss_(a, y), previous_x.T)
			dC_db = dloss_(a, y)
			aux = dloss_(a, y)
			#print('shapes output')
			#print(dC_dW.shape, dC_db.shape, aux.shape)
			return [dC_dW, dC_db, aux]
		else:
			#print(f'delta h: updated_weight matrix{next_model[0].T.shape}\ndelta_values: {all_loss[2].shape}, deriv of Relu: {d_act(a).shape}\ndiference: {all_loss[2]}')
			delta_h = np.dot(next_model[0].T,all_loss[2])*d_act(a)
			dC_dW = np.dot(delta_h, previous_x.T)
			dC_db = delta_h
			#print('shapes hidden')
			#print(delta_h.shape, dC_dW.shape, dC_db.shape)

			return [dC_dW, dC_db, delta_h]


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
		#print('model summary')
		for i in range(epochs):
			for m in range(x.shape[0]): #training the forward on all data
				A = []
				
				#forward pass
				for j in range(len(model)):
					if (j == 0):
						A.append(self.forward(model[j], x[m]))
					else:
						A.append(self.forward(model[j], A[-1]))

					#print(f'{i} iteracao, camada {j} com {model[j][1].shape[0]} valores')
				print('resultado do softmax:', A[-1])
				#backward pass
				#remember to shuffle the entire data
				avg_loss = 0
				for k in np.arange(0, y.shape[0], batch): #updating only batch-size data
					acc = 0
					#loss
					loss_ = getattr(losses, loss)
					avg_loss += loss_(A[-1], y[k])
					print('losss', avg_loss, A[-1])
					
					
					for m in range(y.shape[1]):
						if y[k][m] == 1:
							break
					if (np.argmax(A[-1]) == m):
						acc += 1
						print('YAY')


					all_loss = list()
					for j in range(len(model)-1):
						if j == 0:
							all_loss.append(self.backward(A[-1-j], A[-2-j], y[k], loss, True, model[-j], 0))
						else:
							all_loss.append(self.backward(A[-1-j], A[-2-j], y[k], loss, False, model[-j], all_loss[-1]))
						
						opt_ = getattr(optimizers, opt)
						model[-1-j] = opt_(model[-1-j], all_loss[-1], lr)
				acc /= k
				avg_loss /=k
				print(f'accuracy: {acc}, loss: {avg_loss}')

		return 0,0							