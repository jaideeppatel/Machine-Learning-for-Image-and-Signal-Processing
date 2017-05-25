from scipy.io import wavfile
import numpy as np
import os
import math
from numpy import linalg
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

trs_mat = np.dot(dft,trs_mat)
trs_mat = trs_mat[:513]
trs_mag = np.absolute(trs_mat)

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

trn_mat = np.dot(dft,trn_mat)
trn_mat = trn_mat[:513]
trn_mag = np.absolute(trn_mat)

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

# --------------------------------------- NMF model Signal B_S ------------------------------------------------
B = np.random.normal(0, 0.1, (513, 50))
B_one = np.ones((513, 513))
B = B / np.dot(B_one, B)

O = np.dot(B.T, trs_mag)
O_one = np.ones((50, 50))
O = O / np.dot(O_one, O)

i = 0
epsi = 0.00001
while (True):

    t1 = trs_mag / (np.dot(B, O) + epsi)
    t2 = np.dot(t1, O.T)
    B = B * t2
    B = B / np.dot(B_one, B)

    t1 = trs_mag / (np.dot(B, O) + epsi)
    t2 = np.dot(B.T, t1)
    O = O * t2
    O = O / np.dot(O_one, O)

    Y = np.dot(B, O)
    err = np.linalg.norm(trs_mag - Y)

    i += 1
    if i == 1000:  # Based on the change in the W_s after every iteration decided to break at 1000
        break
B_s = B

# --------------------------------------- NMF model Signal B_N ------------------------------------------------
B = np.random.normal(0, 0.1, (513, 50))
B_one = np.ones((513, 513))
B = B / np.dot(B_one, B)

O = np.dot(B.T, trn_mag)
O_one = np.ones((50, 50))
O = O / np.dot(O_one, O)

i = 0
epsi = 0.001
while (True):

    t1 = trn_mag / (np.dot(B, O) + epsi)
    t2 = np.dot(t1, O.T)
    B = B * t2
    B = B / np.dot(B_one, B)

    t1 = trn_mag / (np.dot(B, O) + epsi)
    t2 = np.dot(B.T, t1)
    O = O * t2
    O = O / np.dot(O_one, O)

    Y = np.dot(B, O)
    err = np.linalg.norm(trn_mag - Y)

    i += 1
    if i == 1000:  # Based on the change in the W_s after every iteration decided to break at 1000
        break
B_n = B

B_sn = np.concatenate((B_s,B_n),axis=1)

O = np.dot(B_sn.T, tex_mag)
O_one = np.ones((100, 100))
O = O / np.dot(O_one, O)

i = 0

while (True):
    t1 = tex_mag / np.dot(B_sn, O)
    t2 = np.dot(B_sn.T, t1)
    O = O * t2
    O = O / np.dot(O_one, O)

    i += 1
    Y = np.dot(B_sn, O)
    err = np.linalg.norm(tex_mag - Y)
    if i == 1000:
        break
O_s = O

Y = np.dot(B_sn,O_s)

phase_x = np.divide(tex_mat,tex_mag)

b_ss = np.multiply(np.dot(B_s, O_s[:50]), phase_x)
b_nn = np.multiply(np.dot(B_n, O_s[50:]), phase_x)

phase_half1 = b_ss
phase_half2 = np.flipud(b_ss[1:512,])
phase_full = np.concatenate((phase_half1, phase_half2), axis=0)

B_sig_real = (phase_full.real)

dft_real = dft.real
dft_t = dft_real.transpose()

B_signal = np.dot(dft_t,B_sig_real)
B_signal = B_signal

org_wave_mat = np.zeros(shape=(159,81920))
for r in range(B_signal.shape[1]):
    org_wave_mat[r, (512 * r):((512 * r) + N)] = B_signal[:, r]

org_wave = org_wave_mat.sum(axis=0) / np.var(org_wave_mat)

# wavfile.write('phase_wave.wav', 16000, org_wave)

Mask = np.divide(np.dot(B_s, O_s[:50]), Y)
signal_out = np.multiply(Mask, tex_mat)
noise_out = np.multiply(1 - Mask, tex_mat)

phase_half1 = signal_out
phase_half2 = np.flipud(signal_out[1:512,])
phase_full = np.concatenate((phase_half1, phase_half2), axis=0)

B_sig_real = (phase_full.real)

dft_real = dft.real
dft_t = dft_real.transpose()

B_signal = np.dot(dft_t,B_sig_real)
B_signal = B_signal

org_wave_mat = np.zeros(shape=(159,81920))
for r in range(B_signal.shape[1]):
    org_wave_mat[r, (512 * r):((512 * r) + N)] = B_signal[:, r]

org_wave_mask = org_wave_mat.sum(axis=0) / np.var(org_wave_mat)

wavfile.write('problem3_mask_wave.wav', 16000, org_wave_mask)

# SNR estimation
org_norm = (tes_org - tes_org.min()) / (tes_org.max() - tes_org.min())
phase_norm = (org_wave - org_wave.min()) / (org_wave.max() - org_wave.min())
mask_norm = (org_wave_mask - org_wave_mask.min()) / (org_wave_mask.max() - org_wave_mask.min())

SNR_ph = 10 * math.log(np.sum(org_norm**2) / np.sum((org_norm - phase_norm)**2),10)
print('SNR Phase:',SNR_ph)

SNR_mask = 10 * math.log(np.sum(org_norm**2) / np.sum((org_norm - mask_norm)**2),10)
print('SNR mask:',SNR_mask)



