# Deds - Deep Learning From Scratch

## Objective
Your simple numpy from scratch deep learning library, created just to keep track of what's happening behind the scenes of TensorFlow. The implementation is similar to Keras. Currently there's only available the Dense format with a few simple functions and the SGD (Stochastic Gradient Descent) optimizer.

## Implementation
For a simple test

```python
python deds.py model=MNIST #mnist
python deds.py model=Wheat #wheat seeds class prediction
```

Or just modify using <b>main.py</b>

## Example at Deds
```python
# input_shape must have (batch, features, 1)
# unlike keras that goes (batch, features, )
from model import Model
NN = Model()

model = NN.Input(10, input_shape=X_train.shape[1], activation='ReLu')
model = NN.Dense(10, 7, model, activation='ReLu')
model = NN.Dense(7, 5, model, activation='ReLu')
model = NN.Output(5, 1, model, activation='Linear')

#train the model categoric labels for now needs to be explicit
loss, accuracy = NN.Train(model, X_train, Y_train, 
	loss='MSE', opt='SGD', epochs=10, batch=8, categoric=False, lr=0.04)
```

## Example at Keras
```python
from keras import layers, models

model = models.Sequential()
model.add(layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(1, activation='linear'))

model.compile(optimizer='sgd', loss='mse', metrics= ['accuracy'])

history = model.fit(
    x=X_train, y=Y_train, 
    epochs=10, batch_size=8
)

```


## Tests

```python
python -m unittest tests/tests_deds.py
```