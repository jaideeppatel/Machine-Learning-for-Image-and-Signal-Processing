import scipy.io as sio
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as nlp

# ----------------------------- Question 2.1

X_Mat = sio.loadmat('X.mat')

A1 = X_Mat['X']
A1_trans = A1.transpose()
A = np.dot(A1,A1_trans) 

X0 = np.array([1,1])
# X0 = X0.transpose()

def get_eigenVector(X0,A):
    while(True):
        AX = np.dot(A,X0)
        # AX = np.around(AX,decimals=3)
        ev = nlp.norm(AX)
        Xi = AX / ev
        Xi = np.round(Xi,decimals=3)
        if (np.array_equal(X0,Xi)):
            break
        X0 = Xi
    return Xi,ev

v1,v_value = get_eigenVector(X0,A)
print("Eigen value and Eigen vector for A:",v_value,v1)

t1 = v1 * v_value
t2 = np.outer(t1,v1)
A_new = A - t2

X0 = np.array([1,1])

v2, v2_value = get_eigenVector(X0,A_new)
print("Second Eigen value and Eigen vector for A:",v2_value,v2)

line = plt.figure(figsize=(10,10))
x = A1[0]
y = A1[1]
plt.plot(x, y, ".")
arr = plt.axes()
# ax.arrow(0,0, 4*e_vec1[0], 4*e_vec1[1], head_width=0.05, head_length=0.1, fc='k', ec='r')
# ax.arrow(0,0, 4*e_vec2[0], 4*e_vec2[1], head_width=0.05, head_length=0.1, fc='k', ec='r')
arr.arrow(0, 0, v1[0], v1[1],head_width=0.05, head_length=0.1, fc='k', ec='r')
arr.arrow(0, 0, v2[0], v2[1],head_width=0.05, head_length=0.1, fc='k', ec='r')
plt.show()


# ------------------------------------------Source Separation -------------------------

# plt.matshow(flute_A,cmap=plt.cm.Blues)
os.chdir('J:\Box Sync\Spring 2017\MLSP\Assignments\Homework #1')
flute_file = sio.loadmat('flute.mat')
flute_A = flute_file['X']

plt.matshow(flute_A)
plt.show()



total_scores = np.sum(flute_A,axis=0)
avg_scores = total_scores / 143

unity = np.ones(shape=(128,128))

a = flute_A - ((np.dot(unity,flute_A))/128)

a_trans = a.transpose()

cov_mat = np.dot(a,a_trans)

X0 = np.ones(shape=(128,1))

flute_vec, flute_value = get_eigenVector(X0,cov_mat)

print("Eigen value and Eigen vector for A:",flute_value,flute_vec)

# v1_trans = v1.transpose()
f1 = flute_vec * flute_value
f2 = np.outer(f1,flute_vec)

cov_mat_new = cov_mat - f2

X0 = np.ones(shape=(128,1))
flute2_vec, flute2_value = get_eigenVector(X0,cov_mat_new)
print("Second Eigen value and Eigen vector for A:",flute2_value,flute2_vec)

b_vec = np.concatenate((flute_vec, flute2_vec),axis=1)
plt.matshow(b_vec)
plt.show()

b_trans = b_vec.transpose()
t_vec = np.dot(b_trans,flute_A)
f_cap = np.dot(b_vec,t_vec)

plt.matshow(f_cap)


from skimage.io import imread
train_image = imread("sgx_train.jpg")
train_image.shape

# Building the matrix for 225 X 185*186
i=0
j=0
x = np.zeros(shape = (225,1))
for i in range(0,186):
    for j in range(0,186):
        temp = train_image[j:(j+15),i:(i+15)]
        t1 = temp.reshape(225,1)
        x = np.concatenate((x,t1),axis=1)
