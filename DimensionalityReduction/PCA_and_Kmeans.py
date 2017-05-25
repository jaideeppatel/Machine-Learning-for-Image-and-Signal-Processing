from scipy.io import wavfile
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
from scipy.stats import mode


input_mat = scipy.io.loadmat('eeg.mat')
# print(input_mat)

y_test = input_mat['y_te']
y_train = input_mat['y_train']

x_test = input_mat['x_te']
x_train = input_mat['x_train']

ctrain_1 = x_train[:,0,:]

ctrain_2 = x_train[:,1,:]

ctrain_3 = x_train[:,2,:]


ctest1 = x_test[:,0,:]

ctest2 = x_test[:,1,:]

ctest3 = x_test[:,2,:]



N = 64
hop = 16

bmann_values = np.blackman(N).transpose()

def find_bmann_values(channel):
    # print(channel.shape)
    zro = np.zeros(shape=(784,))
    zro[:768,] = channel
    channel = zro
    X_list = []
    j = 0
    for i in range(0, channel.shape[0], 48):
        # print(i)
        n_samples = channel[i:(i + 64)]
        j=j+1
        col_vec =  np.multiply(bmann_values,n_samples)
        X_list.append(col_vec)
        if(j>=16):
            break

    bmann_mat = np.array(X_list)
    return bmann_mat

col_list = []

for cols in range(112):

    ch1 = ctrain_1[:,cols]
    bch1 = find_bmann_values(ch1)
    bch1 = bch1.transpose()[:33,:]
    bch1 = bch1[2:7,]
    bch1 = np.reshape(bch1,newshape=(80,))
    # print(bch1.shape)

    ch2 = ctrain_2[:, cols]
    bch2 = find_bmann_values(ch2)
    bch2 = bch2.transpose()[:33, :]
    bch2 = bch2[2:7, ]
    bch2 = np.reshape(bch2, newshape=(80,))
    # print(bch2.shape)

    ch3 = ctrain_3[:, cols]
    bch3 = find_bmann_values(ch3)
    bch3 = bch3.transpose()[:33, :]
    bch3 = bch3[2:7, ]
    bch3 = np.reshape(bch3, newshape=(80,))
    # print(bch3.shape)

    long_vec = np.concatenate((bch1,bch2,bch3),axis=0)

    col_list.append(long_vec)

main_mat = np.array(col_list).transpose()

main_mat_tran = main_mat.transpose()

cov_mat = np.dot(main_mat,main_mat_tran)

eigenval, eigenvec = np.linalg.eig(cov_mat)

plt.plot(eigenval)


def get_accuracy(e,k):

    eg = e
    k_near = k

    new_evec = eigenvec[:eg,:]

    dimred_mat_train = np.dot(new_evec,main_mat).transpose()

    test_collist = []

    for cols in range(28):

        ch1 = ctest1[:,cols]
        bch1 = find_bmann_values(ch1)
        bch1 = bch1.transpose()[:33,:]
        bch1 = bch1[2:7,]
        bch1 = np.reshape(bch1,newshape=(80,))
        # print(bch1.shape)

        ch2 = ctest2[:, cols]
        bch2 = find_bmann_values(ch2)
        bch2 = bch2.transpose()[:33, :]
        bch2 = bch2[2:7, ]
        bch2 = np.reshape(bch2, newshape=(80,))
        # print(bch2.shape)

        ch3 = ctest3[:, cols]
        bch3 = find_bmann_values(ch3)
        bch3 = bch3.transpose()[:33, :]
        bch3 = bch3[2:7, ]
        bch3 = np.reshape(bch3, newshape=(80,))
        # print(bch3.shape)

        longtest_vec = np.concatenate((bch1,bch2,bch3),axis=0)

        test_collist.append(longtest_vec)

    main_mat_test = np.array(test_collist).transpose()

    main_mat_test_tran = main_mat_test.transpose()

    cov_mat_test = np.dot(main_mat_test,main_mat_test_tran)

    eigenvalt, eigenvect = np.linalg.eig(cov_mat_test)

    plt.plot(eigenval)

    new_evect = eigenvect[:eg,:]

    dimred_mat_test = np.dot(new_evect,main_mat_test).transpose()


    result_list = []
    for te in range(28):
        dist_list = {}
        for tr in range(112):
            dist = np.linalg.norm(dimred_mat_test[te,:]-dimred_mat_train[tr,:])
            dist_list[tr]=dist

        newA = sorted(dist_list,key=dist_list.get,reverse=True)[:k_near]
        # print(newA)
        y_elem = [y_train[l][0] for l in newA]
        out_label = mode(y_elem)[0][0]
        result_list.append(out_label)


    true_labels = [y_test[t][0][0] for t in y_test]
    accuracy = len([i for i, j in zip(result_list, true_labels) if i == j])/28
    # print(accuracy)
    return accuracy


for eges in range(10,30,2):
    for k in range(3,12,2):
        acc = get_accuracy(eges,k)
        print('Accuracy for PCs: ',eges,' and Number of NN: ',k,'is: ',acc)

