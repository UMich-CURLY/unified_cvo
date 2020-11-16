import numpy as np
import matplotlib.pyplot as plt

names = ["geometric_cvo", "color_cvo", "semantic_cvo", "gicp", "ndt", "mc_icp"]
colors = ["blue", "green", "red", "purple", "orange", "deepskyblue"]
paths = [
        "/home/rayzhang/code/docker_home/outdoor_cvo/paper/cvo_geometric_img_gpu0_oct23/",
        "/home/rayzhang/code/docker_home/outdoor_cvo/paper/cvo_intensity_img_gpu0_oct25_best/",
 "/home/rayzhang/code/docker_home/outdoor_cvo/paper/cvo_img_semantic_oct26_best/",
"/home/rayzhang/code/docker_home/outdoor_cvo/paper/baselines_stereo/results_gicp_oct28/",
"/home/rayzhang/code/docker_home/outdoor_cvo/paper/baselines_stereo/results_ndt_oct28/",
"/home/rayzhang/code/docker_home/outdoor_cvo/paper/baselines_stereo/results_mc_oct28/"]
#mc_path="/home/rayzhang/code/docker_home/outdoor_cvo/baselines/stereo/results_mc/"
plt.figure(1)
plt.tight_layout()
#f, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=False, sharey=False)
#axs = [ax1, ax2, ax3, ax4]
post_str = ["tl", "rl", "ts", "rs"]

x_axis_unit = [
    "Path Length (m)",
    "Path Length (m)",
    "Speed (m/s)",
    "Speed (m/s)"
]
y_axis_unit = [
    "Translation Error (%)",
    "Rotation Error (deg/m)",
    "Translation Error (%)",
    "Rotation Error (deg/m)"
]
for topic_id in range(len(post_str)):
    curr_topic = post_str[topic_id]
    plt.subplot(411+topic_id)
    #if (topic_id == 0):
    #    ax.legend().draggable()
    for name_id in range(len(names)):
        traj = np.genfromtxt(paths[name_id] +"plot_error/avg_"+curr_topic+".txt" )
        if (topic_id % 2 > 0):
            traj[:, 1] = traj[:, 1] * 180 / 3.1415926
        plt.subplots_adjust(hspace = 0.5)
        if (name_id > 2):
            plt.plot(traj[:, 0], traj[:,1], label=names[name_id],linestyle="--",linewidth=1.0, color=colors[name_id])
        else:
            plt.plot(traj[:, 0], traj[:,1], label=names[name_id],linewidth=1.0, color=colors[name_id])
        plt.xlabel(x_axis_unit[topic_id])
        plt.ylabel(y_axis_unit[topic_id])

#plt.legend(bbox_to_anchor=(0,5,1,0.2),loc='lower left',
#           mode="expand", borderaxespad=0, ncol=4)
plt.legend(loc='lower left',
           mode="expand", borderaxespad=0, ncol=4).set_draggable(True)

plt.show()        
        
    

    
