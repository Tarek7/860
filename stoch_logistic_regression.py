import numpy as np
import pandas as pd
import random
import math
from sklearn.datasets.samples_generator import make_regression 
import matplotlib.pyplot as plt
import pylab
from scipy import stats
from stats import compute_acc
from utils import sigmoid, aug_sigmoid, get_lipschitz_mu
from tqdm import tqdm


def gradient_comp(x, y, lamb, i, w):
	x_i = x[i,:]
	y_i = y[i]
	gradient = float(-y_i) * sigmoid(-y_i * np.dot(x_i, w)) * x_i
	return gradient

def gradient_descent(x_train, y_train, x_test, y_test, alpha, T, lamb):
	train_accuracies = []
	test_accuracies = []

	n = x_train.shape[0]
	p = x_train.shape[1]
	w = np.ones(p)

	for t in tqdm(range(0, T)):
		gradient = 0
		
		i = random.randint(0, n-1)
		gradient = gradient_comp(x_train, y_train, lamb, i, w)
		w = w - alpha * (gradient + 2 * lamb * w)

		train_acc, test_acc = compute_acc(x_train, y_train, x_test, y_test, w)
		train_accuracies.append(train_acc)
		test_accuracies.append(test_acc)

	return w, train_accuracies, test_accuracies


def heavy_ball_descent(x_train, y_train, x_test, y_test, alpha, T, lamb):
	train_accuracies = []
	test_accuracies = []

	n = x_train.shape[0]
	p = x_train.shape[1]
	w = np.ones(p)
	w_prev = w

	L, mu = get_lipschitz_mu(x_train, y_train, lamda, w)
	alpha = 0.8 * 4.0 / (math.sqrt(L) + math.sqrt(mu))**2
	beta = 0.8 * (math.sqrt(L) - math.sqrt(mu))**2 / (math.sqrt(L) + math.sqrt(mu))**2

	for t in tqdm(range(0, T)):
		w_before = w
		i = random.randint(0, n-1)
		gradient = gradient_comp(x_train, y_train, lamb, i, w) + 2 * lamb * w
		w = w - alpha * gradient + beta * (w - w_prev)

		train_acc, test_acc = compute_acc(x_train, y_train, x_test, y_test, w)
		train_accuracies.append(train_acc)
		test_accuracies.append(test_acc)

		if t >= 1:
			w_prev = w_before

	return w, train_accuracies, test_accuracies


def FISTA_descent(x_train, y_train, x_test, y_test, alpha, T, lamb):
	train_accuracies = []
	test_accuracies = []

	n = x_train.shape[0]
	p = x_train.shape[1]
	w = np.ones(p)
	v = w
	u = w
	eta = 0

	for _ in tqdm(range(0, T)):
		i = random.randint(0, n-1)
		gradient = gradient_comp(x_train, y_train, lamb, i, w) + 2.0 * lamb * w

		new_eta = (1. + math.sqrt(1. + 4. * eta**2)) / 2.
		w = v - alpha * gradient
		v = w + (eta - 1.) / float(new_eta) * (w - u)
		eta = new_eta
		u = w

		train_acc, test_acc = compute_acc(x_train, y_train, x_test, y_test, w)
		train_accuracies.append(train_acc)
		test_accuracies.append(test_acc)

	return w, train_accuracies, test_accuracies


def Nesterov_88(x_train, y_train, x_test, y_test, alpha, T, lamb):
	train_accuracies = []
	test_accuracies = []

	n = x_train.shape[0]
	p = x_train.shape[1]
	w = np.ones(p)
	v = w

	for t in tqdm(range(0, T)):
		w_prev = w
		i =random.randint(0, n-1)
		gradient = gradient_comp(x_train, y_train, lamb, i, w) + 2. * lamb * w
		w = v - alpha * gradient
		v = w + (float(t)/(t+3.)) * (w - w_prev)

		train_acc, test_acc = compute_acc(x_train, y_train, x_test, y_test, w)
		train_accuracies.append(train_acc)
		test_accuracies.append(test_acc)

	return w, train_accuracies, test_accuracies	

if __name__ == '__main__':
	x_train=np.load('X_train.npy')
	x_test=np.load('x_test.npy')
	y_train=np.load('y_train.npy')
	y_test=np.load('y_test.npy')

	T = 1500
	lamda = 0
	alpha = 0.005

	w_GD, train_accuracies_GD, test_accuracies_GD = gradient_descent(x_train, y_train, x_test, y_test, alpha, T, lamda)
	w_HB, train_accuracies_HB, test_accuracies_HB = heavy_ball_descent(x_train, y_train, x_test, y_test, alpha, T, lamda)
	w_FISTA, train_accuracies_FISTA, test_accuracies_FISTA = FISTA_descent(x_train, y_train, x_test, y_test, alpha, T, lamda)
	w_N88, train_accuracies_N88, test_accuracies_N88 = Nesterov_88(x_train, y_train, x_test, y_test, alpha, T, lamda)
	

	x = np.arange(1., T + 1., 1.)

	plt.plot(x, train_accuracies_GD, 'r--', test_accuracies_GD, 'r', train_accuracies_N88, 'b--',  test_accuracies_N88, 'b',
				train_accuracies_HB, 'g--', test_accuracies_HB, 'g', train_accuracies_FISTA, '--m', test_accuracies_FISTA, 'm')
	plt.title("T = {}, lambda = {}, alpha = {}".format(T, lamda, alpha))
	plt.show()

