from mpl_toolkits import mplot3d
#%matplotlib inline
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import sys, os

points = np.genfromtxt(sys.argv[1], delimiter=" ")
print(points.shape)

fig = plt.figure()
ax  = fig.gca(projection='3d')
x = points[:,1]
y = points[:, 2]
z = points[:, 3]
ax.plot(x, y, z)
ax.legend()
plt.show()
