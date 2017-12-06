import matplotlib.pyplot as plt
import sys
import seaborn as sns
import numpy as np
import pandas as pd
import random

from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split

sigmoid = lambda x : 1/ (1 + np.exp(-x))

# Genereate data through a mixture of Gaussian.
def gauss_gen(n_samples, n_features):
	n_bins = 2
	center1 = np.ones(n_features) * 5
	center2 = np.zeros(n_features)

	X, y = make_blobs(n_samples = n_samples, n_features = n_features,  centers = [center1, center2], cluster_std=50, center_box = (-15, 15))

	for i in range(n_samples):
		if y[i] == 0:
			y[i] = -1


	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

	print(X_train.shape)
	print(y_train.shape)

	# Write to file for usability.
	np.save('X_train', X_train)
	np.save('X_test', X_test)
	np.save('y_train', y_train)
	np.save('y_test', y_test)


## Multivariate generation.

def multivar_gen(n_samples, n_features):
	np.random.seed(50)
	w = np.matrix(np.random.multivariate_normal([0.0]*n_features, np.eye(n_features))).T
	X = np.matrix(np.random.multivariate_normal([0.0]*n_features, np.eye(n_features), size = n_samples))

	entry = X*w + np.random.normal(0.0, 4.0)
	y = 2 * (np.random.uniform(size = (n_samples, 1)) < sigmoid(entry)) - 1

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

	print(X_train.shape)
	print(y_train.shape)

	np.save('X_train', X_train)
	np.save('X_test', X_test)
	np.save('y_train', y_train)
	np.save('y_test', y_test)

	sys.stdout.write("Generated data successfully\n")



if __name__ == '__main__':
	n_samples = 1000
	n_features = 100

	answer = input("Are you sure? PREVIOUS DATA WILL GO")

	if answer == "yes":
		multivar_gen(n_samples, n_features)
