from scipy import *
import matplotlib.pyplot as plt

def plot_sample(x, y, axis):
	img = x.reshape(96, 96)
	axis.imshow(img, cmap='gray')
	axis.scatter(y[0::2]*48 + 48, y[1::2]*48 + 48, marker='x', s=10)

def create_plot(X, y_pred):
	fig = plt.figure(figsize=(6,6))
	fig.subplots_adjust(
		left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

	for i in range(16):
		ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
		plot_sample(X[i], y_pred[i], ax)
	plt.show()
