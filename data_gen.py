from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split


## Genereate data through a mixture of Gaussian.

n_samples = 500
n_bins = 2
n_features = 50
X, y = make_blobs(n_samples = n_samples, n_features = n_features,  centers = n_bins, cluster_std=5.0, center_box = (-15, 15))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

print(X_train.shape)
print(y_train.shape)

## Write to CSV for usability.
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)

X_train.to_csv('X_train')
X_test.to_csv('X_test')
y_train.to_csv('y_train')
y_test.to_csv('y_test')