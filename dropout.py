from LoadData import *
from lasagne import layers
from lasagne.updates import nesterov_momentum, rmsprop
from nolearn.lasagne import NeuralNet, BatchIterator
import matplotlib.pyplot as plt
import theano
import pickle

def float32(k):
	return np.cast['float32'](k)

#Class for updating network parameters
class AdjustVariable(object):
	def __init__(self, name, start = 0.03, stop=0.001):
		self.name = name
		self.start, self.stop = start, stop
		self.ls = None

	def __call__(self, nn, train_history):
		if self.ls is None:
			self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

		#Get the last epoch
		epoch = train_history[-1]['epoch']
		#Get the new value for the parameter at this epoch
		new_value = float32(self.ls[epoch-1])
		#Set the nn's parameter to this new value
		getattr(nn, self.name).set_value(new_value)

class FlipBatchIterator(BatchIterator):
	#Flip the labels for left and right features when we flip the image
	#horizontally. Left eye becomes right eye, etc.
	flip_indices = [ (0,2), (1,3), (4,8), (5,9), (6,10), (7,11),
			(12,16), (13,17), (14,18), (15,19), (22,24), (23,25) ]

	def transform(self, Xb, yb):
		bs = Xb.shape[0] #Number of samples
		#Flip half of the samples in this batch at random
		indices = np.random.choice(bs, bs/2, replace=False)
		Xb[indices] = Xb[indices, :, :, ::-1] #Flip Horizontal
		
		if yb is not None:
			#Flip all x coordinates horizontally
			yb[indices, ::2] = yb[indices, ::2]* -1
			
			#Swap the indeces of feature labels
			for a, b in self.flip_indices:
				yb[indices, a], yb[indices,b] = (yb[indices, b], yb[indices, a])

		return Xb, yb




net6 = NeuralNet(
	layers=[ #Three layers, one hidden
		('input', layers.InputLayer),
		('conv1', layers.Conv2DLayer),
		('pool1', layers.MaxPool2DLayer),
		('dropout1', layers.DropoutLayer),
		('conv2', layers.Conv2DLayer),
		('pool2', layers.MaxPool2DLayer),
		('dropout2', layers.DropoutLayer),
		('conv3', layers.Conv2DLayer),
		('pool3', layers.MaxPool2DLayer),
		('dropout3', layers.DropoutLayer),
		('hidden4', layers.DenseLayer),
		('dropout4', layers.DropoutLayer),
		('hidden5', layers.DenseLayer),
		('output', layers.DenseLayer),
	],
	#Layer parameters:
	input_shape=(None, 1,96, 96), #96x96 input pixels per batch
	conv1_num_filters=32, conv1_filter_size=(3,3), pool1_ds=(2,2),
	dropout1_p=0.1,
	conv2_num_filters=64, conv2_filter_size=(2,2), pool2_ds=(2,2),
	dropout2_p=0.1,
	conv3_num_filters=128, conv3_filter_size=(2,2), pool3_ds=(2,2),
	dropout3_p=0.1,
	hidden4_num_units=500, #500 hidden units
	dropout4_p=0.1,
	hidden5_num_units=500, #500 hidden units
	output_nonlinearity=None, #output layer uses identity function
	output_num_units=30, #30 target values

	#Optimization method
	update=nesterov_momentum,
	update_learning_rate=theano.shared(float32(0.03)),
	update_momentum=theano.shared(float32(0.9)), #Make these theano shared variables
	#update=rmsprop,
	#update_learning_rate=0.01,
	#update_rho=0.9,
	#update_epsilon=1e-06,

	regression=True, #we are doing regression, not classification
	batch_iterator_train=FlipBatchIterator(batch_size=128),
	on_epoch_finished = [
		AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
		AdjustVariable('update_momentum', start=0.9, stop=0.999),
		],
	max_epochs=10, #number of epochs to train
	verbose=1,
	)

X,y = load2d()
net6.fit(X, y)

#Save the net to a file
import sys
sys.setrecursionlimit(10000)
with open('net6.pickle', 'wb') as f:
	pickle.dump(net6, f, -1)

train_loss = np.array([i["train_loss"] for i in net6.train_history_])
valid_loss = np.array([i["valid_loss"] for i in net6.train_history_])
plt.plot(train_loss, linewidth=3, label="train")
plt.plot(valid_loss, linewidth=3, label="valid")
plt.grid()
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.ylim(1e-3,1e-2)
plt.yscale('log')
plt.show()

X,_ = load2d(test=True)
y_pred = net6.predict(X)

from sklearn.metrics import mean_squared_error
print mean_squared_error(y_pred, y)
from plot_sample import *
create_plot(X, y_pred)
