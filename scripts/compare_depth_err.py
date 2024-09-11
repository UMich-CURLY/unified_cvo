import numpy as np
import sys, os

def compare(f1, f2):
    err1 = np.genfromtxt(f1)
    err2 = np.genfromtxt(f2)

    diff = (err1 - err2) > 0
    print("# of errors decrese from f1 to f2: {} out of total {}".format(diff.sum(), diff.shape))


if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2])
