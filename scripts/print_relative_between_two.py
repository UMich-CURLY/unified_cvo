import numpy as np
from scipy.spatial.transform import Rotation as R
import sys, os
accum_pose_f = sys.argv[1]
start_time = int(sys.argv[2])
end_time = int(sys.argv[3])
def get_relative_pose(poses, t1, t2):
    tails = np.array([[0,0,0,1]])
    p1 = poses[t1].reshape((3,4))
    p2 = poses[t2].reshape((3,4))
    aff1 = np.matrix(np.concatenate((p1, tails), axis=0))
    aff2 = np.matrix(np.concatenate((p2, tails), axis=0))
    return aff1.I * aff2


gtpose = np.genfromtxt(accum_pose_f )


np.set_printoptions(suppress=True)

print ("time {} and {}".format(start_time, end_time ))
pose = (get_relative_pose(gtpose,start_time, end_time))
print(pose)
