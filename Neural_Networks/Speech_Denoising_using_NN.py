from scipy.io import wavfile
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
from scipy.stats import mode

trs = wavfile.read('trs.wav')
trs_mat = trs[1]
trs_org = trs_mat / np.var(trs_mat)


trn = wavfile.read('trn.wav')
trn_mat = trn[1]
trn_org = trn_mat / np.var(trn_mat)


tes = wavfile.read('tes.wav')
tes_mat = tes[1]
tes_org = tes_mat / np.var(tes_mat)

tex = wavfile.read('tex.wav')
tex_mat = tex[1]
tex_org = tex_mat / np.var(tex_mat)

N = 1024

dft = np.zeros(shape=(N, N),dtype=complex)

for f in range(N):
    for n in range(N):
        dft[f][n] = math.cos((2 * math.pi * f)*(n/N)) + ((math.sin((2 * math.pi * f)*(n/N)))* 1j)

def hann_window(n):
    a = 0.5 - (0.5* math.cos((2 * math.pi * n) /(N-1)) )
    return a

han_values = []
for h in range(N):
    val = hann_window(h)
    han_values.append(val)
han_values = np.array(han_values)

#---------------------------------- TRS ----------------------------------------------------
X_list = []
j = 0
for i in range(0, trs_org.shape[0], int(N/2)):
    n_samples = trs_org[i:(i + N)]
    j=j+1
    col_vec =  np.multiply(han_values,n_samples)
    X_list.append(col_vec)
    if j>= 891:
        break

trs_mat = np.array(X_list)
trs_mat = trs_mat.transpose()

trs_spec = np.dot(dft,trs_mat)
trs_mag = trs_spec[:513]
trs_mag = np.absolute(trs_mag)



#---------------------------------- TRN ----------------------------------------------------
X_list = []
j = 0
for i in range(0, trn_org.shape[0], int(N/2)):
    n_samples = trn_org[i:(i + N)]
    j=j+1
    col_vec =  np.multiply(han_values,n_samples)
    X_list.append(col_vec)
    if j>= 891:
        break

trn_mat = np.array(X_list)
trn_mat = trn_mat.transpose()

trn_spec = np.dot(dft,trn_mat)
trn_mag = trn_spec[:513]
trn_mag = np.absolute(trn_mag)


#---------------------------------- Mixture ----------------------------------------------------
x_mat = trs_mag + trn_mag
x_mag = np.absolute(x_mat)


ibm = np.zeros((trs_mag.shape[0],trs_mag.shape[1]))
for i in range(trs_mag.shape[0]):
    for j in range(trn_mag.shape[1]):
        if (trs_mag[i,j] > trn_mag[i,j]):
            ibm[i,j] = 1

x1 = x_mag
bias  = [1] * 891
bias = np.array(bias).reshape((1,891))
x1 = np.concatenate((x1,bias),axis=0)

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


np.random.seed(20)
a1 = np.random.normal(0, 1, 50 * 514)
a1 = a1.reshape((50, 514))

a2 = np.random.normal(0, 1, 513 * 50)
a2 = a2.reshape((513, 50))

alpha = 0.05
error = []
err_old = math.inf
for i in range(1000):
    z1 = np.dot(a1, x1)
    x2 = activation_function(z1)

    z2 = np.dot(a2, x2)
    y_cap = activation_function(z2)

    err = np.square(y_cap - ibm) / 900
    err_new = np.sum(err)
    if (err_new > err_old):
        break
    err_old = err_new

    g2 = (err) * y_cap * (1 - y_cap)
    a2 = a2 - alpha * np.dot(g2, x2.T)

    g1 = np.dot(a2.T, g2) * x2 * (1 - x2)
    a1 = a1 - alpha * np.dot(g1, x1.T)

#---------------------------------- TES ----------------------------------------------------
X_list = []
j = 0
for i in range(0, tes_org.shape[0], int(N/2)):
    n_samples = tes_org[i:(i + N)]
    j=j+1
    col_vec =  np.multiply(han_values,n_samples)
    X_list.append(col_vec)
    if j>= 159:
        break

tes_mat = np.array(X_list)
tes_mat = tes_mat.transpose()

tes_mat = np.dot(dft,tes_mat)
tes_mat = tes_mat[:513]
tes_mag = np.absolute(tes_mat)


#---------------------------------- TEX ----------------------------------------------------
X_list = []
j = 0
for i in range(0, tex_org.shape[0], int(N/2)):
    n_samples = tex_org[i:(i + N)]
    j=j+1
    col_vec =  np.multiply(han_values,n_samples)
    X_list.append(col_vec)
    if j>= 159:
        break

tex_mat = np.array(X_list)
tex_mat = tex_mat.transpose()

tex_mat = np.dot(dft,tex_mat)
tex_mat = tex_mat[:513]
tex_mag = np.absolute(tex_mat)

testx1 = tex_mag
bias  = [1] * 159
bias = np.array(bias).reshape((1,159))
testx1 = np.concatenate((testx1,bias),axis=0)

# Test the input signal

z1 = np.dot(a1,testx1)
x2 = activation_function(z1)

z2 = np.dot(a2,x2)
y_cap_test = activation_function(z2)

y_pred_test = y_cap_test
y_pred_test[np.where(y_pred_test <= 0.5 )] = 0
y_pred_test[np.where(y_pred_test >  0.5 )] = 1

test_output = tex_mat * y_pred_test
full_output = np.vstack((test_output,np.flipud(test_output[1:-1]).conjugate()))


inv_top = np.dot(dft.T,full_output.real)
inv_top = inv_top / np.var(inv_top)

torg_wave_mat = np.zeros(shape=(159,81920))
for r in range(inv_top.shape[1]):
    torg_wave_mat[r, (512 * r):((512 * r) + N)] = inv_top[:, r]

torg_wave = (torg_wave_mat.sum(axis=0)) / 5


wavfile.write('problem1_output.wav', 16000, torg_wave)

sig_norm = (tes_org - np.min(tes_org)) / (np.max(tes_org) - np.min(tes_org))

cleanmix = (torg_wave - np.min(torg_wave)) / (np.max(torg_wave) - np.min(torg_wave))

SNR = 10 * math.log(np.sum(sig_norm**2) / np.sum((sig_norm - cleanmix)**2),10)
print('SNR for Model:',SNR)

