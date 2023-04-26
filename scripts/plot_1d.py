from matplotlib import pyplot as plt
import numpy as np
import csv
from numpy import nonzero
import sys, os


y_arr = np.loadtxt(sys.argv[1], skiprows=1, delimiter = ",")
print(y_arr.shape)
x = np.arange(y_arr.shape[0])
plt.plot(x, y_arr[:,0], 'r', label='ell')
plt.plot(x, y_arr[:,1]/10000, 'g', label='nonzeros')
plt.plot(x, y_arr[:,2], 'b', label='error')
plt.legend()
plt.show()
