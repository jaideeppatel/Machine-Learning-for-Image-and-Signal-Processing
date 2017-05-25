from scipy.io import wavfile
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
from scipy.stats import mode

# read files and generate appropriate dimension vectors

trs = wavfile.read('trs.wav')
# trs_mat = trs[1] / trs[1].var()
trs_mat = trs[1] / 3000
trs_org = trs_mat
print(trs_mat.shape)

trn = wavfile.read('trn.wav')
# trn_mat = trn[1] / trn[1].var()
trn_mat = trn[1] / 3000
trn_org = trn_mat
print(trn_mat.shape)

nmf = wavfile.read('x_nmf.wav')
nmf_mat = nmf[1] / nmf[1].var()
nmf_mat = nmf[1] / 3000
nmf_org = nmf_mat
plt.plot(nmf_org)
print(nmf_mat.shape)


N = 1024

dft = np.zeros(shape=(N, N),dtype=complex)

for f in range(N):
    for n in range(N):
        dft[f][n] = math.cos((2 * math.pi * f)*(n/N)) + ((math.sin((2 * math.pi * f)*(n/N)))* 1j)

print(dft[0,:10])

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
    # print(i+N,j)
    j=j+1
    col_vec =  np.multiply(han_values,n_samples)
    X_list.append(col_vec)
    if(j>=986):
        break

trs_mat = np.array(X_list)
trs_mat = trs_mat.transpose()

trs_spec = np.dot(dft,trs_mat)
trs_spec = trs_spec[:513]
trs_mag = np.absolute(trs_spec)

plt.imshow(trs_mat,aspect='auto',cmap='magma')

#---------------------------------- TRN ----------------------------------------------------

X_list = []
j = 0
for i in range(0, trn_org.shape[0], int(N/2)):
    n_samples = trn_org[i:(i + N)]
    j=j+1
    col_vec =  np.multiply(han_values,n_samples)
    X_list.append(col_vec)
    if(j>=986):
        break

trn_mat = np.array(X_list)
trn_mat = trn_mat.transpose()

trn_spec = np.dot(dft,trn_mat)
trn_spec = trn_spec[:513]
trn_mag = np.absolute(trn_spec)

plt.imshow(trn_mat,aspect='auto',cmap='magma')

#---------------------------------- NMF ----------------------------------------------------

X_list = []
j = 0
for i in range(0, nmf_org.shape[0], int(N/2)):
    n_samples = nmf_org[i:(i + N)]
    j=j+1
    col_vec =  np.multiply(han_values,n_samples)
    X_list.append(col_vec)
    if(j>127):
        break

nmf_mat = np.array(X_list)
nmf_mat = nmf_mat.transpose()

nmf_spec = np.dot(dft,nmf_mat)
nmf_spec = nmf_spec[:513]
nmf_mag = np.absolute(nmf_spec)

plt.imshow(np.absolute(nmf_spec),aspect='auto',cmap='jet')

# --------------------------------------- NMF model Signal W_S ------------------------------------------------

W = np.random.rand(513,30)
H = np.dot(W.transpose(),trs_mag)
err_old = 0
i=0

while (True):
    num = np.dot(trs_mag,H.transpose())
    den = np.dot(W,np.dot(H,H.transpose()))
    W = np.multiply(W, np.divide(num,den))

    num = np.dot(W.transpose(),trs_mag)
    den = np.dot(W.transpose(),np.dot(W,H))
    H = np.multiply(H,np.divide(num,den))

    Y = np.dot(W, H)

    err = abs(np.mean(trs_mag - Y))
    # print('Iteration for W_s', i, 'err is', err)

    i+=1
    if i==1000: # Based on the change in the W_s after every iteration decided to break at 1000
        break
    err_old = err

W_s = W

# --------------------------------------- NMF model Signal W_N ------------------------------------------------

W = np.random.rand(513,30)
H = np.dot(W.transpose(),trn_mag)
err_old = 0
i=0

while (True):
    num = np.dot(trn_mag,H.transpose())
    den = np.dot(W,np.dot(H,H.transpose()))
    W = np.multiply(W, np.divide(num,den))

    num = np.dot(W.transpose(),trn_mag)
    den = np.dot(W.transpose(),np.dot(W,H))
    H = np.multiply(H,np.divide(num,den))

    Y_n = np.dot(W, H)

    err = abs(np.mean(trn_mag - Y_n))
    # print('Iteration for W_n', i, 'err is', err)

    i+=1
    if i==1000: # Based on the change in the W_s after every iteration decided to break at 1000
        break
    err_old = err


W_n = W

W_sn = np.concatenate((W_s,W_n),axis=1)

H = np.random.rand(60,128)
err_old = 0
i=0

while (True):
    num = np.dot(W_sn.transpose(), nmf_mag)
    den = np.dot(W_sn.transpose(), np.dot(W_sn, H))
    H = np.multiply(H, np.divide(num, den))

    i+=1
    if i==1000:
        break

H_s = H
W_S_H = np.dot(W_s, H_s[:30])

angle_x = np.divide(nmf_spec,nmf_mag)

phase_mat_half = np.multiply(W_S_H,angle_x)

phase_flip = np.flipud(phase_mat_half[:512])
phase_mat = np.concatenate((phase_mat_half, phase_flip), axis=0)
phase_mat = phase_mat[:-1]

W_sig_real = (phase_mat.real)

dft_real = dft.real
dft_t = dft_real.transpose()

W_signal = np.dot(dft_t,W_sig_real)
W_signal = W_signal / (N*20)

plt.imshow(W_signal,aspect='auto',cmap='jet')

org_wave_mat = np.zeros(shape=(128,66560))
for r in range(W_signal.shape[1]):
    org_wave_mat[r, (512 * r):((512 * r) + N)] = W_signal[:, r]

org_wave = (org_wave_mat.sum(axis=0))

wavfile.write('nmf_clean.wav', 16000, org_wave)


nmf_org_norm = (nmf_org  - np.min(nmf_org)) / (np.max(nmf_org)-np.min(nmf_org))
org_wave = (org_wave - np.min(org_wave)) / (np.max(org_wave)-np.min(org_wave))

SNR = 10 * math.log(np.sum(nmf_org_norm**2) / np.sum((nmf_org_norm - org_wave)**2),10)
print('SNR for NMF Model:',SNR)

# -------------------------------------------------------------------------------------------------------------

W_S_H = np.dot(W_s, H_s[:30])
W_N_H = np.dot(W_n, H_s[:30])

Mask = np.divide(W_S_H,(W_S_H+W_N_H))


mask_mat = np.multiply(Mask,nmf_spec)

mask_flip = np.flipud(mask_mat[:512])
mask_full = np.concatenate((mask_mat, mask_flip), axis=0)
mask_full= mask_full[:-1]

dft_real = dft.real

dft_t = dft_real.transpose()

mask_mat_inv = np.dot(dft_t, mask_full.real)

mask_mat_inv = mask_mat_inv / (N * 5)

org_wave_mat = np.zeros(shape=(128,66560))
for r in range(mask_mat_inv.shape[1]):
    org_wave_mat[r, (512 * r):((512 * r) + N)] = mask_mat_inv[:, r]

org_wave = (org_wave_mat.sum(axis=0))/2

plt.plot(org_wave)
plt.show()
wavfile.write('nmf_clean_mask.wav', 16000, org_wave)

org_wave = (org_wave - np.min(org_wave)) / (np.max(org_wave)-np.min(org_wave))

SNR = 10 * math.log(np.sum(nmf_org_norm**2) / np.sum((nmf_org_norm - org_wave)**2),10)
print('SNR for mask:',SNR)


