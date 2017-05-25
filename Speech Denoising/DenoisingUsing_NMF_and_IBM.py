from scipy.io import wavfile
import numpy as np
import os
import math
import matplotlib.pyplot as plt

p_wave = wavfile.read('piano.wav')
p_mat = p_wave[1]
p_mat_org = p_wave[1]
plt.plot(p_mat)
N = 1024

dft = np.zeros(shape=(N, N),dtype=complex)

for f in range(N):
    for n in range(N):
        dft[f][n] = math.cos((2 * math.pi * f)*(n/N)) + ((math.sin((2 * math.pi * f)*(n/N)))* 1j)

def hann_window(n):
    a = 0.5 - (0.5* math.cos((2 * math.pi * n) /(N-1)) )
    return a

han_values = []
for h in range(1024):
    val = hann_window(h)
    han_values.append(val)
han_values = np.array(han_values)

X_list = []
j = 0
bound = int(2 * math.floor(p_mat.shape[0] / N)) # Bound limit  = 156 - 1
for i in range(0, p_mat.shape[0], int(N/2)):
    n_samples = p_mat[i:(i + N)]
    j=j+1
    col_vec =  np.multiply(han_values,n_samples)
    X_list.append(col_vec)
    if(j>=bound-1):
        break

P_mat = np.array(X_list)
P_mat = P_mat.transpose()

P_X = np.dot(dft,P_mat)

o_wave = wavfile.read('ocean.wav')
o_mat = o_wave[1]
N = 1024

X_list = []
j = 0
bound = int(2 * math.floor(o_mat.shape[0] / N)) # Bound limit  = 156 - 1
for i in range(0, o_mat.shape[0], int(N/2)):
    n_samples = o_mat[i:(i + N)]
    j=j+1
    col_vec =  np.multiply(han_values,n_samples)
    X_list.append(col_vec)
    if(j>=bound-1):
        break

o_mat = np.array(X_list)
o_mat = o_mat.transpose()

O_X = np.dot(dft,o_mat)



o_abs = np.absolute(O_X)
p_abs = np.absolute(P_X)


Mix_mat = ((o_abs) + p_abs)/(N*5)


Mask = np.divide(np.absolute(p_abs)**2,(np.absolute(p_abs)**2+np.absolute(o_abs)**2))


s_gen = np.multiply(Mask,Mix_mat)
plt.imshow(s_gen,aspect='auto',cmap='jet')

dft_real = dft.real

dft_t = dft_real.transpose()

s_inv = np.dot(dft_t,s_gen)
plt.imshow(s_inv,aspect='auto',cmap='jet')

s_inv_real = s_inv/(N*5)

org_wave_mat = np.zeros(shape=(155,80000))
for r in range(s_inv_real.shape[1]):
    org_wave_mat[r, (512 * r):((512 * r) + N)] = s_inv_real[:, r]

org_wave = (org_wave_mat.sum(axis=0))/2

plt.plot(org_wave)
plt.show()
wavfile.write('piano_clean.wav', 16000, org_wave)

# -----------------------------------------------------------------------------

p_mat_org = (p_mat_org  - np.min(p_mat_org)) / (np.max(p_mat_org)-np.min(p_mat_org))
org_wave = (org_wave - np.min(org_wave)) / (np.max(org_wave)-np.min(org_wave))

SNR = 10 * math.log(np.sum(p_mat_org**2) / np.sum((p_mat_org - org_wave)**2),10)
print('SNR for nonnegative real-values mask:',SNR)

# Ideal Binary Mask Scheme ......
b_list = []
for (x,y),value in np.ndenumerate(P_mat):
    if p_abs[x][y]>o_abs[x][y]:
        b_list.append(1)
    else:
        b_list.append(0)

b_mat = np.array(b_list)
b_mat = np.reshape(b_mat, (1024, 155))

p_new_ibm = np.multiply(b_mat, Mix_mat)
plt.imshow(p_new_ibm,aspect='auto',cmap='jet')

dft_real = dft.real
dft_t = dft_real.transpose()

s_inv = np.dot(dft_t,p_new_ibm)

p_new_ibm_real = s_inv/(N*5)


org_wave_mat_ibm = np.zeros(shape=(155,80000))
for r in range(p_new_ibm_real.shape[1]):
    org_wave_mat_ibm[r, (512 * r):((512 * r) + N)] = p_new_ibm_real[:, r]

org_wave_ibm = (org_wave_mat_ibm.sum(axis=0))

plt.plot(org_wave_ibm)
wavfile.write('piano_clean_ibm.wav', 16000, org_wave_ibm)

org_wave_ibm = (org_wave_ibm - np.min(org_wave_ibm)) / (np.max(org_wave_ibm)-np.min(org_wave_ibm))

SNR = 10 * math.log(np.sum(p_mat_org**2) / np.sum((p_mat_org - org_wave_ibm)**2),10)
print('SNR for Ideal Binary Masks',SNR)