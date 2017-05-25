from scipy.io import wavfile
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import os
import seaborn as sns
import random
import sys
import scipy.spatial.distance as scipydt
import numpy.linalg as nlp

img = plt.imread('IMG_1878.JPG')
plt.rc('axes', **{'grid':False})
plt.imshow(img)

input_img = img
# print(input_img)

# generate 9 random block indexes ...
single_block = 3
r_size = 8
blocks = 9
rand_row = [random.randint(0,input_img.shape[0]-r_size) for i in range(blocks)]

# unpack the dimensions
# x_r1,x_g1,x_b1 = input_img.transpose()

x_r = input_img[:,:,0] # Getting the red components
x_g = input_img[:,:,1]# Getting the blue components
x_b = input_img[:,:,2]# Getting the green components

# Loop for concatenate all the blocks in to a matrix ....

all_x = [x_r,x_g,x_b]
R_mat = np.zeros(shape=(r_size,0))

for x in range(3):
    parts = []
    for y in range(single_block):
        r = random.randint(0,r_size - 8)
        parts.append(all_x[x][r:r+8,:])

    parts = np.array(parts)
    for i in parts:
        R_mat = np.hstack((R_mat,i))

# print(R_mat.shape)

R_temp = R_mat.transpose()

R_temp-= np.mean(R_temp,axis=0)

R_tt = R_temp.transpose()

cov_mat = np.dot(R_tt,R_temp)

eigen_vectors = np.zeros(shape=(r_size,0))

e_values,eigen_vectors = nlp.eig(cov_mat)
W_T = eigen_vectors.transpose()

plt.imshow(W_T,cmap='YlGnBu',interpolation='nearest',)
# plt.colorbar()
plt.show()


# Implementation for 90 blocks ......

# generate 90 random block indexes ...
single_block = 30
r_size = 8
blocks = 90
rand_row = [random.randint(0,input_img.shape[0]-r_size) for i in range(blocks)]

# unpack the dimensions
# x_r1,x_g1,x_b1 = input_img.transpose()

x_r = input_img[:,:,0] # Getting the red components
x_g = input_img[:,:,1]# Getting the blue components
x_b = input_img[:,:,2]# Getting the green components

# Loop for concatenate all the blocks in to a matrix ....

all_x = [x_r,x_g,x_b]
R_mat = np.zeros(shape=(r_size,0))

for x in range(3):
    parts = []
    for y in range(single_block):
        r = random.randint(0,r_size-8)
        parts.append(all_x[x][r:r+8,:])

    parts = np.array(parts)
    for i in parts:
        R_mat = np.hstack((R_mat,i))

# print(R_mat.shape)

R_temp = R_mat.transpose()

R_temp-= np.mean(R_temp,axis=0)

R_tt = R_temp.transpose()

cov_mat = np.dot(R_tt,R_temp)

eigen_vectors = np.zeros(shape=(r_size,0))

e_values,eigen_vectors = nlp.eig(cov_mat)
W_T = eigen_vectors.transpose()

plt.imshow(W_T,cmap='YlGnBu',interpolation='nearest',)
# plt.colorbar()
plt.show()
