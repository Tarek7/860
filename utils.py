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


if __name__ == '__main__':
	print(aug_sigmoid(10))	