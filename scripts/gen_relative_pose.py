import numpy as np
from scipy.spatial.transform import Rotation as R
import sys, os
accum_pose_f = sys.argv[1]
start_time = int(sys.argv[2])
num_frames = int(sys.argv[3])
if len(sys.argv) > 3:
    kitti_or_tartan_or_tum = int(sys.argv[4])
else:
    kitti_or_tartan_or_tum = 0

def xyzq_to_T(xyzq):
    print("xyzq is ", xyzq)
    T = np.eye(4, dtype=float)
    T[0,3] = xyzq[0]
    T[1,3] = xyzq[1]
    T[2,3] = xyzq[2]
    quat = R.from_quat([xyzq[-1], xyzq[-4], xyzq[-3], xyzq[-2]])
    r_mat = quat.as_matrix()
    T[:3,:3] = r_mat
    return T

def get_relative_pose(poses, t1, t2, pose_type):
    tails = np.array([[0,0,0,1]])
    if pose_type == 0:
        # kitti
        p1 = poses[t1].reshape((3,4))
        p2 = poses[t2].reshape((3,4))
        aff1 = np.matrix(np.concatenate((p1, tails), axis=0))
        aff2 = np.matrix(np.concatenate((p2, tails), axis=0))
        return aff1.I * aff2        
    elif pose_type == 1:
        # tartan 
        p1 = xyzq_to_T(poses[t1])
        p2 = xyzq_to_T(poses[t2])
    else:
        # tartan 
        p1 = xyzq_to_T(poses[t1][1:])
        p2 = xyzq_to_T(poses[t2][1:])
    #print("p1 ", p1)
    #print("p2 ", p2)
    # print(np.linalg.inv(p1) @ p2)
    return np.linalg.inv(p1) @ p2



gtpose = np.genfromtxt(accum_pose_f )

pairs = []
for i in range(num_frames):
    pairs.append((start_time+i, start_time+i+1))

np.set_printoptions(suppress=True)

for p in pairs:
    print ("time {} and {}".format(p[0], p[1]))
    pose = (get_relative_pose(gtpose, p[0], p[1], kitti_or_tartan_or_tum))
    print(pose)
    '''
    q = R.from_dcm(pose[:3,:3]).as_quat()
    print("{} {} {} {} {} {} {}".format(
        pose[0,3], pose[1,3], pose[2,3], 
        q[0], q[1], q[2], q[3]
        ))
    '''
    print("================================")
