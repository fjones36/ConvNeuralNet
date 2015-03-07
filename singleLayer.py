from LoadData import *
from lasagne import layers
from lasagne.updates import nesterov_momentum, rmsprop
from nolearn.lasagne import NeuralNet
import matplotlib.pyplot as plt

net1 = NeuralNet(
	layers=[ #Three layers, one hidden
		('input', layers.InputLayer),
		('hidden', layers.DenseLayer),
		('output', layers.DenseLayer),
	],
	#Layer parameters:
	input_shape=(None, 9216), #96x96 input pixels per batch
	hidden_num_units=100, #100 hidden units
	output_nonlinearity=None, #output layer uses identity function
	output_num_units=30, #30 target values

	#Optimization method
	update=nesterov_momentum,
	update_learning_rate=0.01,
	update_momentum=0.9,
	#update=rmsprop,
	#update_learning_rate=0.01,
	#update_rho=0.9,
	#update_epsilon=1e-06,

	regression=True, #we are doing regression, not classification
	max_epochs=100, #number of epochs to train
	verbose=1,
	)

X,y = load()
net1.fit(X, y)

train_loss = np.array([i["train_loss"] for i in net1.train_history_])
valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
plt.plot(train_loss, linewidth=3, label="train")
plt.plot(valid_loss, linewidth=3, label="valid")
plt.grid()
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.ylim(1e-3,1e-2)
plt.yscale('log')
plt.show()

X,_ = load(test=True)
y_pred = net1.predict(X)
from plot_sample import *
create_plot(X, y_pred)
