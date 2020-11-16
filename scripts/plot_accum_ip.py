import numpy as np
import matplotlib.pyplot as plt

names = ["geometric_cvo", "color_cvo", "semantic_cvo", "gicp", "ndt", "mc_icp"]
colors = ["blue", "green", "red", "purple", "orange", "deepskyblue"]
paths = [
        "/home/rayzhang/code/docker_home/outdoor_cvo/paper/cvo_geometric_img_gpu0_oct23/",
        "/home/rayzhang/code/docker_home/outdoor_cvo/paper/cvo_intensity_img_gpu0_oct25_best/",
 "/home/rayzhang/code/docker_home/outdoor_cvo/paper/cvo_img_semantic_oct26_best/",
"/home/rayzhang/code/docker_home/outdoor_cvo/paper/baselines_stereo/results_gicp_oct31/",
"/home/rayzhang/code/docker_home/outdoor_cvo/paper/baselines_stereo/results_ndt/",
"/home/rayzhang/code/docker_home/outdoor_cvo/paper/baselines_stereo/results_mc_oct31/"]
#mc_path="/home/rayzhang/code/docker_home/outdoor_cvo/baselines/stereo/results_mc/"
plt.figure(1)
plt.tight_layout()

curr_seq = "04"

#f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=False, sharey=False)
#axs = [ax1, ax2, ax3, ax4]
for name_id in [0 ,4]:
    traj = np.genfromtxt(paths[name_id] +curr_seq +"_ip.txt" )
    plt.subplots_adjust(hspace = 0.5)
    if (name_id > 2):
        plt.plot(traj[:,0], traj[:,1], label=names[name_id],linestyle="--",linewidth=1.0, color=colors[name_id])
    else:
        plt.plot(traj[:,0], traj[:,1], label=names[name_id],linewidth=1.0, color=colors[name_id])
    plt.xlabel("frame index")
    plt.ylabel(curr_seq)

#plt.legend(bbox_to_anchor=(0,5,1,0.2),loc='lower left',
#           mode="expand", borderaxespad=0, ncol=4)
plt.legend(loc='lower left',
           mode="expand", borderaxespad=0, ncol=4).set_draggable(True)

plt.show()        
        
    

    
