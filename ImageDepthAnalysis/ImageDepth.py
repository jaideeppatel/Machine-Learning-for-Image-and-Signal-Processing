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

img = plt.imread('im0.ppm')
left_img = img

plt.imshow(left_img)

img2 = plt.imread('im8.ppm')
right_img = img2
plt.imshow(right_img)
# plt.savefig('Q2-input_image.png')



rows = left_img.shape[0]
cols = left_img.shape[1]


d_mat = np.zeros(shape=(381,390))
for r in range(rows):
    a = right_img[r,:]
    for pic in range(a.shape[0]-40):
        # ind = find_dist(a[pic],left_img[r,pic:pic+40])
        ind = np.argmin(scipydt.cdist([a[pic]], left_img[r, pic:pic+40], 'euclidean')[0])
        d_mat[r][pic] = ind

len_vec = d_mat.shape[0] * d_mat.shape[1]

d_vec = [int(i) for i in d_mat.reshape(len_vec,)]
uniq = np.unique(d_vec)
cnt = np.bincount(d_vec)

sns.set(style="white", context="talk")
fig,ax1 = plt.subplots(figsize=(10, 10), sharex=True)
sns.barplot(uniq, cnt, palette="Set3", ax=ax1)
plt.savefig('Q2-Histogram.png')

# Code for K means clustering ...................

points_arr = d_vec
points_arr = np.array(points_arr)

clu = 2   # Update the clu value to run for the iven number of clusters....

means = [0] * clu
# means_arr = [random.uniform(0,40) for x in means]
# means_arr = [  3.45256277,  14.4676563 ,36.04473258 , 31.04480168,]
means_arr = [  3.45256277,  36.04473258] # Initialized these 2 mean values to avoid forming empty clusters goind forward...
means_arr = np.array(means_arr)

# print('initial means',means_arr)

assign = [0] * len(d_vec)

total_dist_old = sys.maxsize


def find_dist(pt,means_arr):
    a = abs(means_arr - pt)
    return np.argmin(a)

# find new_means and avg_dist
def find_means(clu,assign):
    new_means = [0] * clu

    for i in range(len(d_vec)):
        new_means[assign[i]] = new_means[assign[i]] + d_vec[i]

    cnt = np.bincount(assign)
    for t in range(cnt.shape[0]):
        if np.isnan(cnt[t]):
            cnt[t] = 1
    new_means = np.divide(new_means,cnt)
    return new_means

total_dist = 0
for x in range(len(d_vec)):
    ret_val = find_dist(d_vec[x],means_arr)
    assign[x] = int(ret_val)
    total_dist = total_dist + abs(means_arr[ret_val] - d_vec[x])

new_means = find_means(clu,assign)
# print('-------',total_dist_old,total_dist)
# print('new means----',new_means)

it=0
# while (abs(total_dist_old - total_dist)>=0.0005):
while (np.equal(np.array(new_means),np.array(means)).all()==False):
    means = new_means
    means_arr = np.array(means)
    total_dist_old = total_dist
    assign = [0] * len(d_vec)

    total_dist = 0
    for x in range(len(d_vec)):
        ret_val = find_dist(d_vec[x], means_arr)
        assign[x] = int(ret_val)
        total_dist = total_dist + abs(means_arr[ret_val] - d_vec[x])

    # print('----------', it, total_dist_old, total_dist)
    new_means = find_means(clu, assign)
    print('new means----', means)
    if (it>=10):
        break


# Replace all values with corresponding means values
d_new = d_vec
for t in range(len(assign)):
    d_new[t] = new_means[assign[t]]

new_image = np.array(d_new)
new_image = new_image.reshape(381,390)

# mx = np.max(new_image)
# mn = np.min(new_image)

# new_image = (new_image - mn) / (mx - mn)
plt.imshow(new_image,cmap='gray')
plt.savefig('Q2-K_Means_Result_img.png')

# ax = sns.heatmap(new_image)

# Code gor GMM .................................

points_arr = d_vec
points_arr = np.array(points_arr)

clu = 2

means = [0] * clu
means_arr = [random.uniform(0,40) for x in means]
means_arr = np.array(means_arr)

# print('initial means',means_arr)

sd = [0] * clu
sd_arr = [random.uniform(20,40) for x in sd]
sd_arr = np.array(sd_arr)

# print('initial sds',sd_arr)

assign = [0] * len(d_vec)
total_dist_old = sys.maxsize

def maha_dist(pt, means_arr,sd_arr,clu):
    a = abs(np.divide((pt - means_arr),sd_arr))
    return np.argmin(a)


total_dist = 0
grp =  [[] for _ in range(clu)]

for x in range(len(d_vec)):
    ret_val = maha_dist(d_vec[x],means_arr,sd_arr,clu)
    assign[x] = int(ret_val)
    grp[int(ret_val)].append(d_vec[x])
    total_dist = total_dist + abs(means_arr[ret_val] - d_vec[x])

# find new_means and avg_dist
def find_means_sd(clu,grp):
    new_means = [sum(x) / len(x) for x in grp]
    square = [[] for _ in range(clu)]
    for t in range(len(grp)):
        x = [x**2 for x in grp[t]]
        square[t] = x

    a = [sum(x) / len(x) for x in square]
    new_sd = np.sqrt(a - (np.multiply(new_means,new_means)))

    return new_means,new_sd

new_means,new_sd = find_means_sd(clu, grp)
# print('new means:',new_means)
# print('new sds:',new_sd)


it=0
# while (abs(total_dist_old - total_dist)>=0.0005):
while (np.equal(np.array(new_means),np.array(means)).all()==False):
    means = new_means
    means_arr = np.array(means)
    sd = new_sd
    total_dist = 0
    grp = [[] for _ in range(clu)]
    assign = [0] * len(d_vec)

    total_dist = 0
    for x in range(len(d_vec)):
        ret_val = maha_dist(d_vec[x], means_arr, sd_arr, clu)
        assign[x] = int(ret_val)
        grp[int(ret_val)].append(d_vec[x])
        total_dist = total_dist + abs(means_arr[ret_val] - d_vec[x])

    # print('----------', it, total_dist_old, total_dist)

    new_means, new_sd = find_means_sd(clu, grp)
    # print('new means:', new_means)
    # 1('new sds:', new_sd)

    if (it>=10):
        break

# Replace all values with corresponding means values
d_new = d_vec
for t in range(len(assign)):
    d_new[t] = new_means[clu -1 -assign[t]]

new_image = np.array(d_new)
new_image = new_image.reshape(381,390)

# mx = np.max(new_image)
# mn = np.min(new_image)
#
# new_image = (new_image - mn) / (mx - mn)

plt.imshow(new_image)
