from scipy.io import wavfile
import numpy as np
import os
import math
import matplotlib.pyplot as plt


# Question 1.2 ----------------------------------------------
# Read wav file ...........
wave = wavfile.read('x.wav')
wave_mat = wave[1]
N = 1600

f_cos = np.zeros(shape=(N, N))
f_sin = np.zeros(shape=(N, N))

for f in range(N):
    for n in range(N):
        ex_cos = math.cos((2 * math.pi * f)*(n/N))
        ex_sin = math.sin((2 * math.pi * f)*(n/N))
        f_cos[f][n] = ex_cos
        f_sin[f][n] = ex_sin


# Question 1.3 -----------------------------------------
def hann_window(n):
    a = 0.5 - (0.5* math.cos((2 * math.pi * n) /(N-1)) )
    return a

han_values = []
for h in range(1600):
    val = hann_window(h)
    han_values.append(val)
han_values = np.array(han_values)

# Create a new wave file for hann window

X_list = []
j = 0
bound = int(2* math.floor(wave_mat.shape[0]/N))
for i in range(0,wave_mat.shape[0],int(N/2)):
    n_samples = wave_mat[i:(i+N)]
    j=j+1
    col_vec =  np.multiply(han_values,n_samples)
    X_list.append(col_vec)
    if(j>=bound):
        break;

X_mat = np.array(X_list)
X_mat = X_mat.transpose()
X_mat = X_mat
# print(X_mat.shape)

plt.imshow(X_mat,aspect='auto',cmap='magma')

real = np.dot(f_cos, X_mat)
img = np.dot(f_sin, X_mat)

X_f = np.sqrt(np.square(real) + np.square(img))

plt.matshow(X_f,aspect='0.05')

real_clean = real
real_clean[199:202, :] = 0
real_clean[1399:1402, :] = 0
real_clean = real_clean / N

img_clean = img
img_clean[199:202, :] = 0
img_clean[1399:1402, :] = 0
img_clean = img_clean / N

X_f_clean = np.sqrt(np.square(real_clean) + np.square(img_clean))

# Question 2.4 -------------------------
fig = plt.figure('Spectrogram',figsize=(10,10))
ax = fig.add_subplot(111)
# ax.set_aspect(0.5)
ax.matshow(X_f_clean, aspect='0.05')
plt.show()

# Question 2.5 -------------------------------

X_f_inv = np.zeros(shape=(N,N))

real_inv = np.dot(f_cos.transpose(), real_clean)
# img_inv = np.dot(f_sin.transpose(), img_clean)

# X_f_inv = np.sqrt(np.square(real_inv) + np.square(img_inv))
X_f_inv = real_inv
X_f_inv = X_f_inv / N

fig = plt.figure('Spectrogram',figsize=(10,10))
ax = fig.add_subplot(111)
ax.matshow(X_f_inv, aspect='0.05')
plt.show()

fig = plt.figure('Spectrogram',figsize=(10,10))
ax = fig.add_subplot(111)
# ax.set_aspect(0.5)
ax.matshow(X_f_clean, aspect='0.05')
plt.show()

org_wave_mat = np.zeros(shape=(78,63488))

for r in range(X_f_inv.shape[1]):
    org_wave_mat[r, (800 * r):((800 * r) + N)] = X_f_inv[:, r]

# print(org_wave_mat.shape)
org_wave = org_wave_mat.sum(axis=0)
# print(org_wave.shape)

plt.plot(org_wave)
# Question 2.4 -------------------------

org_wave = org_wave / 2

wavfile.write('x_clean.wav', 16000, org_wave)


plt.plot(range(wave_mat.shape[0]),org_wave)
plt.show()