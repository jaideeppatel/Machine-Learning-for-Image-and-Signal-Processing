import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from itertools import chain

# Load the concentric.mat file ....
M = scipy.io.loadmat('concentric.mat')
M = M["X"].T
bias  = [1] * 152
bias = np.array(bias).reshape((152,1))
d_mat = np.concatenate((M,bias),axis=1)

# Generate class variables for both the circles ....
class_0 = list(np.zeros((51,)))
class_1 = list(np.ones((101,)))
cs = list(chain(class_0,class_1))
actual_class = np.array(cs)
y = actual_class.reshape((1,152)).T

# Sigmoid and accuracy functions to be used for training ....
def activation_function(z):
    sigmoid_value = 1 / (1 + np.exp(-z))
    return sigmoid_value

def accuracy(y,pred):
    same = 0
    for i in range(len(y)):
        if y[i,0] == pred[i]:
            same = same+1
    return same/len(pred)


# np.random.seed(20)
wt_old = np.random.normal(0, 1, 9)
wt_old = wt_old.reshape((3, 3))
print("Initial weights and bias at layer 1:", wt_old)
wt2_old = np.random.normal(0, 1, 3)
wt2_old = wt2_old.reshape((3, 1))
print("Initial weights and bias at layer 2:", wt2_old)
alpha = 0.05
error = []

for i in range(100000):

    sigmoid_value1 = activation_function(np.dot(d_mat, wt_old))
    sigmoid_value2 = activation_function(np.dot(sigmoid_value1, wt2_old))

    error.append(np.sum(abs(sigmoid_value2 - y)))
    err = (y - sigmoid_value2) * 2

    delta_2 = err * (sigmoid_value2 * (1 - sigmoid_value2))
    wt2_old = wt2_old + alpha * np.dot(sigmoid_value1.T, delta_2)

    delta_1 = np.dot(delta_2, wt2_old.T) * sigmoid_value1 * (1 - sigmoid_value1)
    wt_old = wt_old + alpha * d_mat.T.dot(delta_1)

    # Accuracy determination code...
    y_cap = [1 if x >= 0.5 else 0 for x in list(sigmoid_value2)]
    acc = accuracy(y, y_cap)

    if acc == 1:
        print("Neural Network Converges at iteration number: ", i)
        break

print("Final weights and bias at layer 1 are:", wt_old)
print("Final weights and bias at layer 2 are:", wt2_old)

# Out of curiosity to see the behabiour of the linear error during the training process
# I have ploted the error at every iteration ... It is an interesting graph ...
plt.plot(error)
plt.show()
