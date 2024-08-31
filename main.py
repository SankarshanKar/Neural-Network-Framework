from data.datasets import sine_data
from neural_network import Model
from neural_network.accuracy import Accuracy_Regression
from neural_network.activation import Activation_Linear, Activation_ReLU
from neural_network.layer import Layer_Dense
from neural_network.loss import Loss_MeanSquaredError
from neural_network.optimizer import Optimizer_Adam

model = Model()

X, y = sine_data()

model.add(Layer_Dense(1, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 1))
model.add(Activation_Linear())

model.set(
    loss=Loss_MeanSquaredError(),
    optimizer=Optimizer_Adam(learning_rate=0.005, decay=1e-3),
    accuracy=Accuracy_Regression()
)

model.finalize()

model.train(X, y, epochs=10000, print_every=100)
