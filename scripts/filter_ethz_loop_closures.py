import numpy as np
import sys, os

def load_lc(file_name, gt_pose_file, out_lc_file_name):
    correlation = np.genfromtxt(file_name, delimiter=' ')
    poses = np.genfromtxt(gt_pose_file, delimiter=' ')
    print("correlation shape: ",correlation.shape)
    lc_pairs = []
    counter = 0
    with open(out_lc_file_name, 'w') as f:
        for i in range(correlation.shape[0]):
            pose_i = np.eye(4)
            pose_i[:3, :] = poses[i].reshape(3,4)
            for j in range(correlation.shape[1]):
                
                if correlation[i,j] > 0.7 and i != j  and abs(i-j) > 7:
                    pose_j = np.eye(4)
                    pose_j[:3, :] = poses[j].reshape(3,4)
                    print("pose {} to {} is \n".format(i,j), np.linalg.inv(pose_i) @ pose_j)
                    
                    f.write("{} {}\n".format(i, j))
                    #print("{} {}\n".format(i,j))
                    counter+=1
    print(counter)
                    

if __name__ == "__main__":
    load_lc(sys.argv[1], sys.argv[2], sys.argv[3])
