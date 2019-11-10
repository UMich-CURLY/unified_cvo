import numpy as np
from scipy.spatial.transform import Rotation as R
import sys, os
accum_pose_f = sys.argv[1]
start_time = int(sys.argv[2])

def get_relative_pose(poses, t1, t2):
    tails = np.array([[0,0,0,1]])
    p1 = poses[t1].reshape((3,4))
    p2 = poses[t2].reshape((3,4))
    aff1 = np.matrix(np.concatenate((p1, tails), axis=0))
    aff2 = np.matrix(np.concatenate((p2, tails), axis=0))
    return aff1.I * aff2


gtpose = np.genfromtxt(accum_pose_f )

pairs = []
for i in range(50):
    pairs.append((start_time+i, start_time+i+1))

np.set_printoptions(suppress=True)

for p in pairs:
    print ("time {} and {}".format(p[0], p[1]))
    pose = (get_relative_pose(gtpose, p[0], p[1]))
    print(pose)
    '''
    q = R.from_dcm(pose[:3,:3]).as_quat()
    print("{} {} {} {} {} {} {}".format(
        pose[0,3], pose[1,3], pose[2,3], 
        q[0], q[1], q[2], q[3]
        ))
    '''
    print("================================")
