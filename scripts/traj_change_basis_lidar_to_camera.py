import numpy as np
from scipy.spatial.transform import Rotation as R
import sys

def T_traj_lidar_frame_to_cam_frame(pose_file, new_pose_file):
    poses = np.genfromtxt(pose_file)
    r = R.from_euler('xyz', [0, -90, 90], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3,:3] = r
    T_inv = np.linalg.inv(T)
    for i in range(poses.shape[0]):
        pose = np.eye(4)
        pose[:3, :] = poses[i, :].reshape((3,4))
        pose = T @ pose @ T_inv
        #pose = pose @ T
        poses[i, :] = pose[:3, :].reshape((1,12))
    np.savetxt(new_pose_file, poses)


if __name__ == "__main__":
    T_traj_lidar_frame_to_cam_frame(sys.argv[1], sys.argv[2])
        
        
        
        
        
