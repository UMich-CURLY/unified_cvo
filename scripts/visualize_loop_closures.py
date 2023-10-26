import numpy as np
from scipy.spatial.transform import Rotation as sciR
import sys, os
from matplotlib import pyplot as plt

def plot_loop_edges_on_kitti_traj(traj_file, lc_g2o_file):
    poses = np.genfromtxt(traj_file, delimiter=' ')
    plt.scatter(poses[:, 3], poses[:, 11], c='b', s=0.1)
    
    with open(g2o_file, 'r') as f:
        for line in f:
            print("=================================")
            print(line)
            strs = line.split()
            if (strs[0] != "EDGE_SE3:QUAT" ):
                continue
            f1 = int(strs[1])
            f2 = int(strs[2])
            pose1 = np.eye(4)
            x1, y1 = poses[f1, 3], poses[f1, 11]
            x2, y2 = poses[f2, 3], poses[f2, 11]
            plt.arrow(x1, y1, (x2-x1), y2-y1, width=0.1, color='r')

    plt.show()
    
    


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
                    
def print_global_registration_err(g2o_file, gt_file):
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

def get_R_t_err(ba_f1_to_f2, gt_f1_to_f2):
    ba_error_pose = np.linalg.inv(ba_f1_to_f2) @ gt_f1_to_f2
    #print("ba_error is \n", ba_error_pose)
    t_err = np.linalg.norm(ba_error_pose[:3, 3])
    R_err = np.linalg.norm(sciR.from_matrix(ba_error_pose[:3,:3]).as_rotvec(degrees=True))
    return R_err, t_err

def get_relative_pose(pose_array, f1,  f2):
    ba_f1 = np.eye(4)
    ba_f1[:3,:] = pose_array[f1].reshape((3,4))
    ba_f2 = np.eye(4)
    ba_f2[:3,:] = pose_array[f2].reshape((3,4))
    ba_f1_to_f2 = np.linalg.inv(ba_f1) @ ba_f2
    return ba_f1_to_f2
    
                
def print_ba_err(g2o_file, gt_file, ba_file, tracking_file):
    gt_poses = np.genfromtxt(gt_file)
    ba_poses = np.genfromtxt(ba_file)
    tracking_poses = np.genfromtxt(tracking_file)
    np.set_printoptions(precision=4)

    with open(g2o_file, 'r') as f:
        total_tracking_R_err, total_tracking_t_err, total_ba_R_err, total_ba_t_err = 0,0,0,0
        counter = 0
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

            tracking_f1_to_f2 = get_relative_pose(tracking_poses, f1, f2)
            ba_f1_to_f2 = get_relative_pose(ba_poses, f1, f2)
            gt_f1_to_f2 = get_relative_pose(gt_poses, f1, f2)        
            
            
            print("global reg is \n",pose_f1_to_f2)
            print("tracking is \n", tracking_f1_to_f2)
            print("ba is \n", ba_f1_to_f2)
            print("gt is \n", gt_f1_to_f2)

            init_R_err, init_t_err = get_R_t_err(pose_f1_to_f2, gt_f1_to_f2)
            print("global_reg_error is \n", np.linalg.inv(pose_f1_to_f2) @ (gt_f1_to_f2))
            if init_R_err > 0.25:
                print("Warning: global_reg translation error > 0.25, err = ", init_R_err)
            if init_t_err > 30:
                print("Warning: global_reg rotation error > 30 degree, err = ", init_t_err)

            tracking_R_err, tracking_t_err = get_R_t_err(tracking_f1_to_f2, gt_f1_to_f2)
            print("tracking_error is \n", np.linalg.inv(tracking_f1_to_f2) @ (gt_f1_to_f2))
            if tracking_R_err > 0.25:
                print("Warning: global_reg translation error > 0.25, err = ", tracking_R_err)
            if tracking_t_err > 30:
                print("Warning: global_reg rotation error > 30 degree, err = ", tracking_t_err)
                

            ba_R_err, ba_t_err = get_R_t_err(ba_f1_to_f2, gt_f1_to_f2)
            print("ba_error is \n", np.linalg.inv(ba_f1_to_f2) @ (gt_f1_to_f2))
            if init_R_err > 0.25:
                print("Warning: ba translation error > 0.25, err = ", ba_R_err)
            if init_t_err > 30:
                print("Warning: ba rotation error > 30 degree, err = ", ba_t_err)

            total_tracking_R_err += tracking_R_err
            total_tracking_t_err += tracking_t_err
            total_ba_R_err += ba_R_err
            total_ba_t_err += ba_t_err
            counter+= 1
            print("After BA, error reduces from {},{} to {},{}".format(tracking_R_err, tracking_t_err, ba_R_err, ba_t_err))

        print("After BA, mean error reduces from {},{} to {},{}".format(total_tracking_R_err / counter, total_tracking_t_err/ counter, total_ba_R_err  / counter, total_ba_t_err / counter))
                
                
                
if __name__ == "__main__":
    load_lc(sys.argv[1], sys.argv[2], sys.argv[3])
    print_registration_err(sys.argv[1], sys.argv[2])

    #########################################################
    # ba_error on loop closure
    # code:
    #g2o_file = sys.argv[1]
    #gt_file = sys.argv[2]
    #ba_file = sys.argv[3]
    #tracking_file = sys.argv[4]
    #print_ba_err(g2o_file, gt_file, ba_file, tracking_file)
    #########################################################
    # plot loop closures
    #traj_file = sys.argv[1]
    #g2o_file = sys.argv[2]
    #plot_loop_edges_on_kitti_traj(traj_file, g2o_file)
