import numpy as np
import math
from utils import sigmoid


def compute_acc(x_train, y_train, x_test, y_test, w):
	n = len(x_train)
	m = len(x_test)

	train_acc = 0
	for i in range(n):
		x_i = x_train[i]
		y_i = y_train[i]
		prob = sigmoid(np.dot(w, x_i))
		if prob >= 0.5:
			y = 1
		else:
			y = -1
		train_acc += (y != y_i)
	
	train_acc = float(train_acc) / float(n)

	test_acc = 0
	for i in range(m):
		x_i = x_test[i]
		y_i = y_test[i]
		prob = sigmoid(np.dot(w, x_i))
		if prob >= 0.5:
			y = 1
		else:
			y = -1
		test_acc += (y != y_i)
	
	test_acc = float(test_acc) / float(m)

	return train_acc, test_acc


def compute_acc_SVM(x_train, y_train, x_test, y_test, w):
	n = len(x_train)
	m = len(x_test)

	train_acc = 0
	for i in range(n):
		x_i = x_train[i]
		y_i = y_train[i]
		y = -1 if np.dot(w, x_i) > 0 else 1
		train_acc += (y != y_i)
	
	train_acc = float(train_acc) / float(n)

	test_acc = 0
	for i in range(m):
		x_i = x_test[i]
		y_i = y_test[i]
		prob = sigmoid(np.dot(w, x_i))
		y = -1 if np.dot(w, x_i) > 0 else 1
		test_acc += (y != y_i)
	
	test_acc = float(test_acc) / float(m)

	return train_acc, test_acc	
