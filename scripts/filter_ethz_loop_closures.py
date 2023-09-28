import numpy as np
import sys, os

def load_lc(file_name, out_lc_file_name):
    correlation = np.genfromtxt(file_name, delimiter=' ')
    print("correlation shape: ",correlation.shape)
    lc_pairs = []
    counter = 0
    with open(out_lc_file_name, 'w') as f:
        for i in range(correlation.shape[0]):
            for j in range(correlation.shape[1]):
                if correlation[i,j] > 0.8 and i != j  and abs(i-j) > 1:
                    f.write("{} {}\n".format(i, j))
                    print("{} {}\n".format(i,j))
                    counter+=1
    print(counter)
                    

if __name__ == "__main__":
    load_lc(sys.argv[1], sys.argv[2])
