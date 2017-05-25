from scipy.io import wavfile
import numpy as np
import os
import math
import matplotlib.pyplot as plt


# Read the input files

x_ica1 = wavfile.read('x_ica_1.wav')
xica1_mat = x_ica1[1]
xica1_mat_org = xica1_mat

x_ica2 = wavfile.read('x_ica_2.wav')
xica2_mat = x_ica2[1]
xica2_mat_org = xica2_mat

x_ica3 = wavfile.read('x_ica_3.wav')
xica3_mat = x_ica3[1]
xica3_mat_org = xica3_mat

x_ica4 = wavfile.read('x_ica_4.wav')
xica4_mat = x_ica4[1]
xica4_mat_org = xica4_mat


X_mat = np.array(([xica1_mat],[xica2_mat],[xica3_mat],[xica4_mat]))
X_mat = X_mat[:,0,:]

means = np.mean(X_mat,axis=1)
# print(means)

X_means =  X_mat - means[:, np.newaxis]

cov_mat = np.dot(X_means,X_means.transpose())

eigenval, eigenvec = np.linalg.eig(cov_mat)

W = np.divide(eigenvec,np.sqrt(eigenval))

Z_mat = np.dot(W.T, X_means)

np.cov(Z_mat)

I_mat = (np.identity(4) * 42292)

Y = np.ones(shape=(4,42292))
Y_old = Y
W = eigenvec



p = 0.00001
i=0
error_old = 0

while i<200:  # After experimenting on the wav files and the sounds heard from them, the changes in the error values
# I have decided to break the loop at this point as the error remains constant after this...

    G_y = np.tanh(Y)

    F_y = np.power(Y, 3)

    w_delta = np.dot(I_mat - np.dot(G_y,F_y.transpose()),W)
    W  = W + w_delta*p

    Y = np.dot(W,Z_mat)

    error = abs(np.sum(Y_old - Y))
    # print('Iteration',i,'err is',error)
    Y_old = Y

    # if (err-error_old)<0.0000001:
    #     break
    # if error>error_old:
    #     p = p*10
    # else:
    #     p = p/10
    i+=1


# print(Y.shape)

wavfile.write('x_ica1_clean.wav', 16000, Y[0])

wavfile.write('x_ica2_clean.wav', 16000, Y[1])

wavfile.write('x_ica3_clean.wav', 16000, Y[2])

wavfile.write('x_ica4_clean.wav', 16000, Y[3])


