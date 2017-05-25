import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from mpl_toolkits.mplot3d import Axes3D
import random

# Import the MDS_pdist file ....
M = scipy.io.loadmat('MDS_pdist')
M = M['L']

# Perform row mean and col mean operation ....
M_row_mean = np.mean(M,axis=1)
M_tilda = M - M_row_mean
M_tilda_col_mean = np.mean(M_tilda,axis=0)
W = M_tilda - M_tilda_col_mean
W = -0.5*W

# Generate the Eigenvalue and Wigen vectors ....
eigenval, eigenvec = np.linalg.eig(W)
eigen_values = eigenval.real
eigen_vectors = eigenvec.real
eval_mat = np.zeros((996,996))
np.fill_diagonal(eval_mat,eigen_values)

# Plot the points in a 2-Dimensional space ....
coord = np.dot(eigen_vectors,eval_mat)
x_axis = coord[:,0]
y_axis = coord[:,1]
z_axis = coord[:,2]

z = np.sqrt(x_axis.real**2 + y_axis.real**2)
plt.scatter(x_axis.real, y_axis.real, s=80, c=z_axis, marker=">")
plt.show()

# Use rotational matrix to get a better plot ...
r = np.array([[0,-1],[1,0]])
xy = np.array([x_axis,y_axis])
new_xy = np.dot(r,xy)
nx = new_xy[0]
ny = new_xy[1]

z = np.sqrt(nx.real**2 + ny.real**2)
plt.scatter(nx.real, ny.real, s=80, c=z, marker=">")
plt.show()

# For experimentation plotted the points in a 3-Dimensional space ....
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax = Axes3D(fig)
ax.scatter(x_axis, y_axis, z_axis, c = z_axis,s=20)
plt.show()