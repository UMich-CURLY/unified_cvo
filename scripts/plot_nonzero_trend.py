from matplotlib import pyplot as plt
import numpy as np
import csv
from numpy import nonzero
import sys, os

with open(sys.argv[1], 'r', newline='') as file:
    reader = csv.reader(file, delimiter=',')
    x_arr = []
    y_arr = []
    for row in reader:
        # print(float(row[0]))
        x_arr.append(float(row[0]))
        y_arr.append(float(row[1]))

plt.plot(y_arr)
plt.show()
