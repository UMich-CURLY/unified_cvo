import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import sys, os
import pdb
from scipy import linalg
from numpy import linalg as LA
from liegroups import SO3
def rotationError(pose_error):
    a = pose_error[0,0];
    b = pose_error[1][1];
    c = pose_error[2][2];
    d = 0.5*(a+b+c-1.0);
    return np.arccos(max(min(d,1.0),-1.0))


def translationError(pose_error):
    dx = pose_error[0][3]
    dy = pose_error[1][3]
    dz = pose_error[2][3]
    return np.sqrt(dx*dx+dy*dy+dz*dz)


ell = np.genfromtxt(sys.argv[1])

transform = np.genfromtxt(sys.argv[2])
gt_frame = int(sys.argv[3])
gt = np.genfromtxt(sys.argv[4])
ip = np.genfromtxt(sys.argv[5])
dist = np.genfromtxt(sys.argv[6])
zero1 = np.array([0,0,0,1.0])
gt_now = gt[gt_frame,:].reshape((3,4))
gt_now = (np.vstack([gt_now, zero1]))
gt_next = gt[gt_frame+1,:].reshape((3,4))
gt_next = np.vstack([gt_next, zero1])
gt_curr = LA.inv(gt_now) @ gt_next
r_err = np.zeros(ell.size)
t_err = np.zeros(ell.size)
for i in range(ell.size):
    t_i = transform[i, :].reshape((3,4))
    t_i = np.vstack([t_i, zero1])

    #e = 
    error = linalg.inv(gt_curr) @ t_i
    #error =  gt.I * t_i 
    r_err[i] = LA.norm(SO3.log(SO3.from_matrix(linalg.inv(t_i[:3,:3]) @ gt_curr[:3,:3], normalize=True)))         #rotationError(error)
    #if r_err[i] > 0:
    #    print(r_err[i])
    t_err[i] = LA.norm(t_i[:3, 3] - gt_curr[:3, 3])

 
    

iters = np.arange(ell.size)
#dist_line, = plt.plot(iters, dist, label="dist")
ell_line, = plt.plot(iters,ell, label="ell", linestyle='dashed')
ip_line, = plt.plot(iters, ip * 1e2, label='inner_product')
rerr_line, = plt.plot(iters, r_err * 10, label="rotation error", linestyle="dotted")
terr_line, = plt.plot(iters, t_err, label="translation error", linestyle="dotted")
#w1_l, = plt.plot(iters, w1, label="w1")
#w2_l, = plt.plot(iters, w2, label="w2")
#w3_l, = plt.plot(iters, w3, label="w3")
#v1_l, = plt.plot(iters, v1, label="v1")
#v2_l, = plt.plot(iters, v2, label="v2")
#v3_l, = plt.plot(iters, v3, label="v3")
#plt.legend(handles=[ell_line, dist_line, step_line, rerr_line,terr_line ])
#plt.legend(handles=[rerr_line,terr_line,w1_l, w2_l, w3_l, v1_l, v2_l, v3_l ])
plt.legend(handles=[ell_line, ip_line,  rerr_line,terr_line])
plt.show()

