import numpy as np
import pandas as pd
import random
import math
from sklearn.datasets.samples_generator import make_regression 
import matplotlib.pyplot as plt
import pylab
from scipy import stats
from stats import compute_acc
from utils import sigmoid, aug_sigmoid
from tqdm import tqdm

def viz_data(methods_list, T):

	tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

	for i in range(len(tableau20)):    
		r, g, b = tableau20[i]    
		tableau20[i] = (r / 255., g / 255., b / 255.)  

	plt.figure(figsize=(12, 9))  

	x = np.arange(1., T + 1., 1.)

	legend_list = ['Gradient Descent', 'Nesterov 88', 'Heavy Ball', 'FISTA']

	index = 0
	for counter, value in enumerate(methods_list): 
		if counter % 2 == 0:
			plt.plot(x, value, lw=2.0, color=tableau20[counter], label = legend_list[index])
			index +=1
		else:
			plt.plot(x, value, lw=2.0, color=tableau20[counter - 1], linestyle='--')

	legend = plt.legend(loc='upper right', shadow=False, bbox_to_anchor=(1.11, 1.005))

	frame = legend.get_frame()
	frame.set_facecolor('0.90')

	plt.title("Batch with Early Stopping Methods (- - : train | -- : test)", {'fontsize':17}, loc="center") 

	plt.show()

	plt.savefig("batch - iterative - logistic regression", bbox_inches="tight")

