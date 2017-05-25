import scipy
import numpy as np
import os
import numpy.random as nprand
import matplotlib.pyplot as plt
from skimage.viewer import ImageViewer

os.chdir('J:\Box Sync\Spring 2017\MLSP\Assignments\Homework #1')
from skimage.io import imread

train_image = imread("sgx_train.jpg")/255

train_pic = imread("sgx_train.jpg")
train_pic = ImageViewer(train_pic)
train_pic.show()

# Building the matrix for 225 X 185*186
i=0
j=0

x = []
for i in range(0,186):
    for j in range(0,186):
        temp = train_image[j:(j+15),i:(i+15)]
        t1 = temp.reshape(225,)
        x.append(t1)
    # print(i)

# print(len(x))
X = np.array(x).transpose()

# np.random.seed(232)
f = np.random.rand(225,)
f_trans = f.transpose() / np.linalg.norm(f)
ft_X = np.dot(f_trans,X)
# print('ftx',ft_X)

s_image = imread("sg_train.jpg")/255
s_image_view = ImageViewer(s_image)
s_image_view.show()

y = []
j=7

for j in range(7,193):
    temp2 = s_image[7:193,j]
    y.append(list(temp2))

s_list = []
for item in y:
    s_list.extend(item)

s_vec = np.array(s_list)

def g_f_X(ft_X):
    return (1 / (1 + np.exp(-1*ft_X)))

def g_dash_X(ft_X):
    return ((g_f_X(ft_X)) * (1 - (g_f_X(ft_X))))

g_ft_X = g_f_X(ft_X)
gdash_ft_X = g_dash_X(ft_X)

diff = s_vec - g_ft_X
diff_trans = diff.transpose()
err_old = np.dot(diff,diff_trans) / 34596
eeta = 3

i=0
while (err_old > 0.05):

    i=i+1
    delta_f = (2 / 34596) * (np.dot(X,(np.multiply(diff,gdash_ft_X))))
    f_new = f + (eeta * delta_f)

    f_trans = f_new.transpose() / np.linalg.norm(f_new)
    ft_X = np.dot(f_trans,X)

    g_ft_X = g_f_X(ft_X)
    gdash_ft_X = g_dash_X(ft_X)

    diff = s_vec - g_ft_X
    diff_trans = diff.transpose()
    err_new = np.dot(diff,diff_trans) / 34596
    print(err_new)
    if err_new > err_old:
        # eeta = (eeta/1.5) * (err_new - err_old)
        eeta = (eeta/1.5)
        f_new = f
    else:
        f = f_new
    err_old = err_new
    print(i)
    if i==8000:
        break;

filter = f_new.reshape(15,15)
fig = plt.figure('Filter',figsize=(10,10))
ax = fig.add_subplot(111)
ax.set_aspect(1)
ax.matshow(filter)
plt.show()

# Read the sgx_test file and add the filter to it ....--------------------
test = imread("sgx_test.jpg")/255
test_image = ImageViewer(test)
test_image.show()

i=0
j=0
test_mat = []
for i in range(0,186):
    for j in range(0,186):
        temp = test[j:(j+15),i:(i+15)]
        t1 = temp.reshape(225,)
        test_mat.append(t1)
    # print(i)

# print(len(x))
test_mat = np.array(test_mat).transpose()

# -------------------------------------------------------------------------
train_new_image = np.dot(f_new.transpose() / np.linalg.norm(f_new),X)
train_new_image = train_new_image.reshape(186,186)
train_new_image = train_new_image.transpose()
train_result_pic = ImageViewer(train_new_image)
train_result_pic.show()


new_image = np.dot(f_new.transpose() / np.linalg.norm(f_new),test_mat)
new_image = new_image.reshape(186,186)
new_image = new_image.transpose()
result_pic = ImageViewer(new_image)
result_pic.show()


