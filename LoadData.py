import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

FTRAIN = '~/data/kaggle-facial-keypoint-detection/training.csv'
FTEST = '~/data/kaggle-facial-keypoint-detection/test.csv'

def load(test=False, cols=None):
	'''Loads data from FTEST if test is True otherwise from FTRAIN.
	Pass a list of cols if you are only interested in a subset of the 
	target columns.
	'''
	fname = FTEST if test else FTRAIN
	df = read_csv(os.path.expanduser(fname)) #Load pandas fataframe

	#Image column has pixel values separated by space
	#convert to values to numpy arrays
	df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

	if cols: #Get a subset of columns
		df = df[list(cols) + ['Image']]

	print(df.count()) #Prints the number of values for each column
	df = df.dropna() #drop all rows that have missing values

	#Stack array to make a verical array each row is an image
	X = np.vstack(df['Image'].values)/255. #scale pixel values to 0-1
	X = X.astype(np.float32)

	if not test: #Only FTRAIN has target columns
		y = df[df.columns[:-1]].values
		y = (y-48)/48. #scale target coordinates -1 to 1
		X,y = shuffle(X, y, random_state=42) #shuffle train data.
		#random_state - Control the shuffling for reproducible behavior
		y = y.astype(np.float32)
	else:
		y = None

	return X, y

if __name__ == "__main__":
	X,y = load()
	print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
		X.shape, X.min(), X.max()))
	print("Y.shape == {}; Y.min == {:.3f}; Y.max == {:.3f}".format(
		y.shape, y.min(), y.max()))
