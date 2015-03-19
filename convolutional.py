from LoadData import *
from lasagne import layers
from lasagne.updates import nesterov_momentum, rmsprop
from nolearn.lasagne import NeuralNet
import matplotlib.pyplot as plt

net2 = NeuralNet(
	layers=[ #Three layers, one hidden
		('input', layers.InputLayer),
		('conv1', layers.Conv2DLayer),
		('pool1', layers.MaxPool2DLayer),
		('conv2', layers.Conv2DLayer),
		('pool2', layers.MaxPool2DLayer),
		('conv3', layers.Conv2DLayer),
		('pool3', layers.MaxPool2DLayer),
		('hidden4', layers.DenseLayer),
		('hidden5', layers.DenseLayer),
		('output', layers.DenseLayer),
	],
	#Layer parameters:
	input_shape=(None, 1,96, 96), #96x96 input pixels per batch
	conv1_num_filters=32, conv1_filter_size=(3,3), pool1_ds=(2,2),
	conv2_num_filters=64, conv2_filter_size=(2,2), pool2_ds=(2,2),
	conv3_num_filters=128, conv3_filter_size=(2,2), pool3_ds=(2,2),
	hidden4_num_units=500, #500 hidden units
	hidden5_num_units=500, #500 hidden units
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
	max_epochs=10, #number of epochs to train
	verbose=1,
	)

X,y = load2d()
net2.fit(X, y)

import cPickle as pickle
with open('net2.pickle', 'wb') as f:
    pickle.dump(net2, f, -1)

train_loss = np.array([i["train_loss"] for i in net2.train_history_])
valid_loss = np.array([i["valid_loss"] for i in net2.train_history_])
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
