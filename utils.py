import numpy as np
import math

def sigmoid(x):  
    return math.exp(-np.logaddexp(0, -x))

def aug_sigmoid(x):
	num = -x
	denom = 0
	if x >= 39:
		denom = 0
		return math.exp(num)
	else:
		denom = 2 * math.log(1 + math.exp(-x))
		return math.exp(num - denom)

def get_lipschitz_mu(x, y, lamda, w):
	n = x.shape[0]
	p = x.shape[1]

	# matrix = np.zeros((p,p))
	# for i in range(n):
	# 	x_i = x[i,:]
	# 	x_im = x_i.reshape((-1, 1))
	# 	x_it = x_im.T
	# 	y_i = y[i]
	# 	matrix += np.dot(x_im, x_it) * aug_sigmoid(y_i * np.dot(x_i, w))
	# matrix = (1. / n)  * matrix + 2 * lamda *  np.eye(p)	

	matrix = (1./n) * (x.T.dot(x)) + 2. * lamda * np.eye(p)	
	eigenvalues = np.linalg.eig(matrix)[0]
	L = max(eigenvalues)
	mu = min(eigenvalues)
	return L, mu		


if __name__ == '__main__':
	print(aug_sigmoid(10))	