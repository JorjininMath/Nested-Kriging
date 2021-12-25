from fastpredict_noise_func import fast_LOOerror
import numpy as np
import math
import pandas as pd
import time
from sklearn.cluster import KMeans

def r(x):
    y = 0.05 + 0.2*(1 + np.sin(2*x)) / (1 + np.exp(-0.2*x))
    y = y ** 2
    return y


"f(x) = sinc(x) = sin(pi*x) / (pi*x)"


def f(x):
    y = math.sin(math.pi * x) / (math.pi * x)
    return y


"""Initialization"""
n = 500
N = 5
q = 2000
d = 1

"We use random sample to get design and predict points"
np.random.seed(1)
x = np.random.uniform(-10, 10, q)
np.random.seed(2)
X = np.random.uniform(-10, 10, n)

"Note that we want to predict f(x) without noise, so we only need variance on design points"
"since f(x) = sin(x) / x, we need to avoid  x = 0"
y_x = np.zeros(q, dtype='float64')
y_X = np.zeros(n, dtype='float64')
for i in range(q):
    if x[i] != 0:
        y_x[i] = f(x[i])
    else:
        y_x[i] = 1

for i in range(n):
    if X[i] != 0:
        y_X[i] = f(X[i])
    else:
        y_X[i] = 1
"noise term"
y_noise = np.zeros(n, dtype='float64')
y_x_noise = np.zeros(q, dtype='float64')
var_noise = r(X)

for i in range(n):
    np.random.seed(2*i)
    y_noise[i] = np.random.normal(0, r(X)[i])
for i in range(q):
    np.random.seed(i)
    y_x_noise[i] = y_x[i] + np.random.normal(0, r(x)[i])

"Use k-means to get clusters, which includes the information of gp and gp_size"
"random_state is the random seed to control results"
"use reshape(-1, 1) to convert X to col vector"
cluster = KMeans(n_clusters=N).fit(X.reshape(-1, 1))
gp = cluster.labels_
"np.bincount() to count the size of each group (this is for non-negative int)"
gpsize = np.bincount(gp)

"kernel parameters"
var = 3
covtype = 'gauss'
param = np.array([43.0], dtype='float64')

"index"
q_test = 20
index = np.zeros(n, dtype='int32')
index[np.random.choice(n, q_test, replace=False)] = 1
# index = np.ones(n, dtype='int32')

"reshape x, X"
x = x.reshape(-1, 1)
X = X.reshape(-1, 1)

"""Apply algorithm"""
"here q is the sum of index"
q_index = sum(index)
t1 = time.time()
Result = fast_LOOerror(X, y_X, n, d, gp, gpsize, N, index, q_index, var_noise,
                   y_noise, covtype, var, param, nuggetfactor=1.001)
t2 = time.time()
print('run time: ', t2 - t1, 's')

"test"
a = Result['LOOerror']
mse = np.average(Result['LOOerror'] ** 2)

