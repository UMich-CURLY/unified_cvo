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

w1=np.genfromtxt("../w1.txt")
w2=np.genfromtxt("../w2.txt")
w3=np.genfromtxt("../w3.txt")
v1=np.genfromtxt("../v1.txt")
v2=np.genfromtxt("../v2.txt")
v3=np.genfromtxt("../v3.txt") 


ell = np.genfromtxt(sys.argv[1])
print(ell.shape)
dist  = np.genfromtxt(sys.argv[2])

transform = np.genfromtxt(sys.argv[3])
print(transform.shape)
gt_frame = int(sys.argv[5])

gt = np.genfromtxt(sys.argv[4])
zero1 = np.array([0,0,0,1.0])
step = np.genfromtxt(sys.argv[6])
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
ell_line, = plt.plot(iters,ell, label="ell", linestyle='dashed')
dist_line, = plt.plot(iters,dist, label="dist", linestyle='solid')
step_line, = plt.plot(iters,step, label="step")
rerr_line, = plt.plot(iters, r_err, label="rotation error", linestyle="dotted")
terr_line, = plt.plot(iters, t_err, label="translation error", linestyle="dotted")
#w1_l, = plt.plot(iters, w1, label="w1")
#w2_l, = plt.plot(iters, w2, label="w2")
#w3_l, = plt.plot(iters, w3, label="w3")
#v1_l, = plt.plot(iters, v1, label="v1")
#v2_l, = plt.plot(iters, v2, label="v2")
#v3_l, = plt.plot(iters, v3, label="v3")
#plt.legend(handles=[ell_line, dist_line, step_line, rerr_line,terr_line ])
#plt.legend(handles=[rerr_line,terr_line,w1_l, w2_l, w3_l, v1_l, v2_l, v3_l ])
plt.legend(handles=[ell_line, step_line, dist_line, rerr_line,terr_line ])
plt.show()

