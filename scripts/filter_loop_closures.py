import numpy as np
from scipy.spatial.transform import Rotation as sciR
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
                    
def print_registration_err(g2o_file, gt_file):
    gt_poses = np.genfromtxt(gt_file)
    np.set_printoptions(precision=4) 
    with open(g2o_file, 'r') as f:
        for line in f:
            print("=================================")
            print(line)
            strs = line.split()
            if (strs[0] != "EDGE_SE3:QUAT" ):
                continue
            f1 = int(strs[1])
            f2 = int(strs[2])
            x,y,z,qx, qy, qz, qw = strs[3:10]
            pose_f1_to_f2 = np.eye(4)
            pose_f1_to_f2[:3,:3] = sciR.from_quat([qx, qy, qz, qw ]).as_matrix()
            pose_f1_to_f2[:3, 3] = np.array([x,y,z])

            gt_f1 = np.eye(4)
            gt_f1[:3,:] = gt_poses[f1].reshape((3,4))
            gt_f2 = np.eye(4)
            gt_f2[:3,:] = gt_poses[f2].reshape((3,4))
            
            print("pred is \n",pose_f1_to_f2)
            print("gt is \n", np.linalg.inv(gt_f1) @ gt_f2)
            error_pose = np.linalg.inv(pose_f1_to_f2) @ (np.linalg.inv(gt_f1) @ gt_f2)
            print("error is \n", np.linalg.inv(pose_f1_to_f2) @ (np.linalg.inv(gt_f1) @ gt_f2))
            if np.linalg.norm(error_pose[:3, 3]) > 0.25:
                print("Warning: translation error > 0.25")
            if np.linalg.norm(sciR.from_matrix(error_pose[:3,:3]).as_rotvec(degrees=True)) > 30:
                print("Warning: rotation error > 30 degree")
    
if __name__ == "__main__":
    #load_lc(sys.argv[1], sys.argv[2], sys.argv[3])
    print_registration_err(sys.argv[1], sys.argv[2])
