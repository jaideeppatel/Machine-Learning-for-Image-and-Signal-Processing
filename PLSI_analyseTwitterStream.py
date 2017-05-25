from scipy.io import wavfile
import numpy as np
import os
import math
from numpy import linalg
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
from scipy.stats import mode


matdata = scipy.io.loadmat('twitter.mat')

xtr = matdata['Xtr']
xte = matdata['Xte']
yte = matdata['YteMat']
ytr = matdata['YtrMat']

# Generate NMF model for Xtr

B = np.random.normal(0, 0.1, (891, 50))
B_one = np.ones((891, 891))
B = B / np.dot(B_one, B)


O = np.dot(B.T, xtr)
O_one = np.ones((50, 50))
O = O / np.dot(O_one, O)

i = 0
epsi = 0.001
while (True):

    t1 = xtr / (np.dot(B, O) + epsi)
    t2 = np.dot(t1, O.T)
    B = B * t2

    B = B / np.dot(B_one, B)

    t1 = xtr / (np.dot(B, O) + epsi)
    t2 = np.dot(B.T, t1)
    O = O * t2

    O = O / np.dot(O_one, O)

    Y = np.dot(B, O)
    err = np.linalg.norm(xtr - Y)

    i += 1
    if i == 1000:  # Based on the change in the B_s after every iteration decided to break at 2000
        break
B_tr = B
O_tr = O

# Generate NMF model for Xte .. only updating O
B = B_tr
O = np.random.normal(0, 0.1, (50, 193))
O_one = np.ones((50, 50))
O = O / np.dot(O_one, O)

i = 0
epsi = 0.001
while (True):

    t1 = xte / (np.dot(B, O) + epsi)
    t2 = np.dot(B.T, t1)
    O = O * t2

    O = O / np.dot(O_one, O)

    Y = np.dot(B, O)
    err = np.linalg.norm(xte - Y)

    i += 1
    if i == 1000:  # Based on the change in the W_s after every iteration decided to break at 1000
        break
O_te = O

bias = np.ones((1,773))
x1 = np.concatenate((O_tr,bias),axis=0)
x1 = x1.T

def softmax(z):
    softmax_value =  (z / (np.sum(z,axis=1)[:,None] + 0.000001))
    return softmax_value


alpha = 0.001
i = 0
a1 = np.random.normal(0, 0.1, 51 * 3)
a1 = a1.reshape((51, 3))
err_list = []
while (True):
    i += 1
    z1 = np.dot(x1, a1)
    z1 = np.exp(z1)
    y_cap = softmax(z1)

    err = - (ytr.T * np.log(y_cap))
    esum = np.sum(np.abs(err))

    g1 = (y_cap - ytr.T)
    a1 = a1 - (alpha * np.dot(x1.T, g1))
    if (i == 50000):
        break

pred = np.argmax(y_cap,axis=1)
actual = np.argmax(ytr.T,axis=1)
correct = pred == actual
bincnt = np.bincount(correct)
acc = bincnt[1] / np.sum(bincnt)
print("Training Accuracy is:",acc)

bias = np.ones((1,193))
x1_test = np.concatenate((O_te,bias),axis=0)
x1_test = x1_test.T

z1 = np.dot(x1_test,a1)
z1 = np.exp(z1)
y_cap_test = softmax(z1)

test_pred = np.argmax(y_cap_test,axis=1)
test_actual = np.argmax(yte.T,axis=1)
test_correct = test_pred == test_actual
test_bincnt = np.bincount(test_correct)
test_acc = test_bincnt[1] / np.sum(test_bincnt)
print("Test Accuracy is:",test_acc)

