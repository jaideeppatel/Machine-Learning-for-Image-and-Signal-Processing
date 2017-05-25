import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from sklearn.metrics.pairwise import euclidean_distances
from itertools import chain
from mpl_toolkits.mplot3d import Axes3D
import random

# Load the concentric.mat file ....
M = scipy.io.loadmat('concentric.mat')
M = M['X']
z = np.sqrt(M[0]**2 + M[1]**2)
plt.scatter(M[0],M[1], s=80, c = z, marker="o")
plt.show()
mat = M.T

dist = euclidean_distances(mat, mat)
dist  = np.square(dist)

# Perform row mean and col mean operations ....
d_row_mean = np.mean(dist,axis=1)
d_tilda = dist - d_row_mean
d_tilda_col_mean = np.mean(d_tilda,axis=0)
W = d_tilda - d_tilda_col_mean
W = -0.5*W;
w_rbf = np.exp(- W / 0.1 )

# Generate eigen values and eigen vectors ....
eigenval, eigenvec = np.linalg.eig(w_rbf)
eval_mat = np.zeros((152,152))
np.fill_diagonal(eval_mat,eigenval.real)

coord = np.dot(eigenvec,eval_mat)
x_axis = coord[:,0]
x_axis = x_axis.real
y_axis = coord[:,1]
y_axis = y_axis.real
z_axis = coord[:,2]
z_axis = z_axis.real

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax = Axes3D(fig)
ax.scatter(x_axis, y_axis, z_axis, c = z_axis,s=20)
plt.show()

# Generate class variables for both the circles ....
class_0 = list(np.zeros((51,)))
class_1 = list(np.ones((101,)))
cs = list(chain(class_0,class_1))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax = Axes3D(fig)
ax.scatter(x_axis, y_axis, z_axis, c = cs,s=20)
plt.show()

# prepare the input data with the bias for the perceptron ....
bias  = [1] * 152
x_y_z = [list(x_axis),list(y_axis),list(z_axis),bias]
d_mat = np.array(x_y_z)

# Sigmoid and accuracy functions to be used for training ....
def activation_function(z):
    sigmoid_value = 1 / (1 + np.exp(-z))
    return sigmoid_value

def activation_function_derivation(z):
    value = np.multiply(activation_function(z), (1 - activation_function(z)))
    return value

def accuracy(cs,pred):
    same = 0
    for i in range(len(cs)):
        if cs[i] == pred[i]:
            same = same+1
    return same / len(cs)


s = np.random.normal(0, 0.1, 4)
wt = s.reshape((1, 4))
wt_old = wt
alpha = 0.025
n = d_mat.shape[1]
print("Initial Weights and bias are",wt_old)
actual_class = np.array(cs)
actual_class = actual_class.reshape((1, 152))
error = []

for i in range(100000):
    i += 1
    z = np.dot(wt_old, d_mat)
    sigmoid_value = activation_function(z)

    difference = sigmoid_value - actual_class
    error.append(np.sum(abs(difference)))
    err = 0.5 * np.dot(difference, difference.T)

    delta_1 = np.multiply(difference, activation_function_derivation(z))
    delta_2 = alpha * (err / n) * np.dot(delta_1, d_mat.T)
    wt_old = wt_old - delta_2

    # Accuracy determination code...
    y_cap = [1 if x >= 0.5 else 0 for x in list(sigmoid_value)[0]]
    acc = accuracy(cs, y_cap)

    if (acc == 1):
        print("The perceptron Converges at iteration number: ", i)
        break

print("Final weights and the bias are:", wt_old)

