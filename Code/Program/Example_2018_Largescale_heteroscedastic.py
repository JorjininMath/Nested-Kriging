from fastpredict_noise_func import fast_pred_noise
import matplotlib.pyplot as plt
import numpy as np
import math
import time
from sklearn.cluster import KMeans
import pickle


"""
We choose the example in 'Large-scale Heteroscedastic Regression via
Gaussian Process' to test our model
"""

"""
Parameters in this example: 
number of design points: 500
number of subsets: 5
f(x): sin(x) + epsilon(x), epsilon(x) ~ N(0, r(x))
r(x): 0.05 + 0.2(1 +sin(2x)) / (1 + exp(-0.2x))

Use k-means method to partition design points into 5 disjoint subsets
"""

"Construct function"


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
q = 1000
d = 1

"We use random sample to get design and predict points"
data1 = pickle.load(open('C:/Users/Jorjin/Desktop/Research 1st/Nested Kriging/Code_Python/data.txt', 'rb'))
x = data1['x'].T
X = data1['X'].T
# np.random.seed(1)
# x = np.random.uniform(-10, 10, q)
# np.random.seed(2)
# X = np.random.uniform(-10, 10, n)

"Note that we want to predict f(x) without noise, so we only need variance on design points"
"since f(x) = sin(pi * x) / (pi * x), we need to avoid  x = 0"
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
"noise-free test"
y_noise_free = np.zeros(n, dtype='float64')
var_noise_free = np.zeros(n, dtype='float64')
y_noise = np.zeros(n, dtype='float64')
y_x_noise = np.zeros(q, dtype='float64')
var_noise = r(X)
# c1 = np.random.uniform(0, 0.15, n)
# c2 = np.random.uniform(0, 0.15, q)
# var_noise = c1
for i in range(n):
    np.random.seed(2*i)
    y_noise[i] = np.random.normal(0, r(X)[i])
    # y_noise[i] = np.random.normal(0, c1[i])
for i in range(q):
    np.random.seed(i)
    y_x_noise[i] = y_x[i] + np.random.normal(0, r(x)[i])
    # y_x_noise[i] = y_x[i] + np.random.normal(0, c2[i])

"Use k-means to get clusters, which includes the information of gp and gp_size"
"random_state is the random seed to control results"
"use reshape(-1, 1) to convert X to col vector"
cluster = KMeans(n_clusters=N).fit(X.reshape(-1, 1))
gp = cluster.labels_
"np.bincount() to count the size of each group (this is for non-negative int)"
gpsize = np.bincount(gp)

"kernel parameters"
var = 15
covtype = 'gauss'
param = np.array([1.5], dtype='float64')

"reshape x, X"
x = x.reshape(-1, 1)
X = X.reshape(-1, 1)

# "We replace cluster with unordered cluster to see what the group choice influence performance"
# gp1 = np.zeros(500, dtype='int32')
# for i in range(100):
#     for j in range(5):
#         gp1[5 * i + j] = j
# gpsize1 = np.array([100, 100, 100, 100, 100])

"""Apply algorithm"""
t1 = time.time()
Result = fast_pred_noise(X, y_X, n, d, gp, gpsize, N, x, q, var_noise,
                   y_noise, covtype, var, param, nuggetfactor=1.0001)
t2 = time.time()
print('run time: ', t2 - t1, 's')

"Error and 95% confidence interval"
m_A = Result['m_A']
v_A = np.sqrt(Result['v_A'])
UB = m_A + 1.96 * v_A
LB = m_A - 1.96 * v_A
mse = np.sum((y_x - m_A)**2) / q


"Figure"
"We need to rearrange array to let them be ordered such that we can use plot"
"We use argsort() to sort it with x-axis"

Data = np.array([x[:, 0], m_A, UB, LB, y_x])
Data = Data[:, Data[0].argsort()]

colors = np.array(['purple', 'green', 'blue', 'yellow', 'brown'])

fig, ax = plt.subplots(figsize=(12, 12))
ax.plot(Data[0], Data[1], color='red', linewidth=2, label='model')
ax.plot(Data[0], Data[2], color='black', linewidth=2, label='LB&UB')
ax.plot(Data[0], Data[3], color='black', linewidth=2)
ax.plot(Data[0], Data[4], color='purple', linewidth=2, label='real')
# ax.scatter(x, y_x_noise, c='blue', marker='+', s=50)
ax.scatter(X, y_X + y_noise, c=colors[gp], marker='+', s=50)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Result of noise-model')
ax.legend()
fig.show()

