import numpy as np
import sys, os

def sesync_pose_to_kitti(pose_f, result_f):
    poses = np.genfromtxt(pose_f)
    num_poses = poses.shape[1] // 4
    t_all = poses[:, :num_poses]
    R_all = poses[:, num_poses:]
    
    with open(result_f, 'w') as f:
        for  i in range(num_poses):
            pose = np.zeros((3, 4))
            pose[:, :3] = R_all[:, i*3:(i*3+3)]
            pose[:, 3] = t_all[:, i]
            pose = np.reshape(pose, (12,))
            for j in range(pose.size):
                if (j < 11):
                    f.write("{} ".format(pose[j]))
                else:
                    f.write("{}\n".format(pose[j]))

if __name__ == "__main__":
    sesync_pose_f = sys.argv[1]
    out_kitti_pose_f = sys.argv[2]
    sesync_pose_to_kitti(sesync_pose_f, out_kitti_pose_f)
