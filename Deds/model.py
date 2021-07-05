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
			all_lost = list()
			#dc_dw
			dc_dz_o = (a - y)
			dz_dw_o = a_t0.T #previous activation
			dc_dw_o = np.dot(dc_dz_o, dz_dw_o)/y.shape[0]

			#dc_db
			dc_db_o = dc_dz_o/y.shape[0]
			return [dc_dz_o, dc_dw_o, dc_db_o]

		else:
			#dc_dw
			dc_dz_t1 = all_loss[0]
			dz_t1_da = W_t1
			da_dz = d_act_(z)
			dc_dz = np.dot((dc_dz_t1).T, dz_t1_da).T *da_dz
			dz_dw = a_t0.T
			dc_dw = np.dot(dc_dz, dz_dw)/y.shape[0]

			#dc_db
			dc_db = dc_dz/y.shape[0]
			return [dc_dz, dc_dw,dc_db]


	def summary(self, model):
		print(f'| Total Number of Layers: {len(model)} |')
		for i in range(len(model)):
			inputs = model[i][0].shape[1]
			outputs = model[i][0].shape[0]

			print('| layer {} with {} inputs and {} outputs neurons |'.format(i+1, 
				inputs, outputs))

	def Input(self, neurons, input_shape, activation):
		#random weights and bias between -0.5 to 0.5
		np.random.seed(23)
		weights = np.random.rand(neurons, input_shape) - 0.5
		bias = np.random.rand(neurons, 1) - 0.5
		return [[weights, bias, activation]]

	def Dense(self, pr_neurons, next_neurons, model, activation):
		np.random.seed(23)
		weights = np.random.rand(next_neurons, pr_neurons) - 0.5
		bias = np.random.rand(next_neurons, 1) - 0.5
		model.append([weights, bias, activation])
		return model

	def Output(self, pr_neurons, next_neurons, model, activation):
		np.random.seed(23)
		weights = np.random.rand(next_neurons, pr_neurons) - 0.5
		bias = np.random.rand(next_neurons, 1) - 0.5
		model.append([weights, bias, activation])
		return model

	def Train(self, model, x, y, loss, opt, epochs, batch, categoric, lr=0.04,
			  momentum=True, gamma=0.95):
		
		#optimizer with momentum
		if not momentum:
			m = False
			gamma = 0
		else:
			m = True
			g=0.95

		l = []
		ac = []
		
		#print summary
		self.summary(model)

		for i in range(epochs):
			avg_loss = 0
			acc = 0
			count = 0
			samp = np.random.randint(0, y.shape[0], size=y.shape[0]//batch)
			#samp = np.arange(0, y.shape[0], batch) #batch sample size
			for k in samp:
				A = list()
				
				count += 1

				#forward pass
				for j in range(len(model)):
					if (j == 0):
						A.append(self.forward(model[j], x[k]))
					else:
						A.append(self.forward(model[j], A[-1][2]))

				#backward pass
				#loss
				loss_ = getattr(losses, loss)
				avg_loss += (loss_(A[-1][2], y[k])).mean()
				
				if categoric:
					if (np.argmax(A[-1][2]) == np.argmax(y[k])):
						acc += 1

				else:
					if (int(A[-1][2][0]) == int(y[k])):
						acc += 1


				all_loss = list()
				for j in range(len(model)):
					if j == 0:
						all_loss.append(self.backward(A[-1-j], model[-1-j][2], y[k], loss, True, model[-j], [0,0,0]))
					else:
						all_loss.append(self.backward(A[-1-j], model[-1-j][2], y[k], loss, False, model[-j], all_loss[-1]))

				if momentum:
					if count == 1:
						momentum_ = [[lr*all_loss[j][1], lr*all_loss[j][2]] for j in range(len(model))]
					else:
						momentum_ = [[gamma*momentum_[j][0] + lr*all_loss[j][1],
								    	gamma*momentum_[j][1] + lr*all_loss[j][2]] for j in range(len(model))]

				#update params
				opt_ = getattr(optimizers, opt)
				model = opt_(model, momentum_) 
			acc /= count
			avg_loss /= count

			print(f'epoch: {i+1}, accuracy: {acc}, loss: {avg_loss}')
			l.append(avg_loss)
			ac.append(acc)

		return model, l, ac

	def Evaluate(self, model, x, y, categoric):
		results = list()
		precision = 0
		for k in range(len(x)):
			for j in range(len(model)):
				if (j == 0):
					results.append(self.forward(model[j], x[k]))
				else:
					results.append(self.forward(model[j], results[-1][2]))

			if categoric:
				if (np.argmax(results[-1][2]) == np.argmax(y[k])):
					precision += 1

			else:
				if (int(results[-1][2][0]) == int(y[k])):
					precision += 1


		print(f'Network got {precision/len(y)} right')
		return precision