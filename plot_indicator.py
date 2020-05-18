import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import sys, os
import pdb
import glob
from scipy import linalg
from numpy import linalg as LA
#from liegroups import SO3
from pathlib import Path

def rotationError(pose_error):
    a = pose_error[0,0]
    b = pose_error[1][1]
    c = pose_error[2][2]
    d = 0.5*(a+b+c-1.0)
    return np.arccos(max(min(d,1.0),-1.0))


def translationError(pose_error):
    dx = pose_error[0][3]
    dy = pose_error[1][3]
    dz = pose_error[2][3]
    return np.sqrt(dx*dx+dy*dy+dz*dz)

seq_names = sorted(glob.glob("/home/cel/PERL/code/DockerFolder/media/Samsung_T5/lyft/lyft_kitti/train/*"))
kitti_seq = "01"
gt_frame = 100
zero1 = np.array([0,0,0,1.0])
dataset = "kitti"

plt.rcParams.update({'font.size': 20})
plt.rcParams.update({'lines.markersize': 3})
# a = ['0', '10', '10', '10']
# b = ['0', '10', '11', '09']
a = ['05']
b = ['10']
my_dpi = 100
decay_rate = "08"

if dataset == "kitti":
    folder_name = '/home/cel/Pictures/indicator/kitti/'
    Path(folder_name).mkdir(parents=True, exist_ok=True)
    final_t_error_list = []
    final_r_error_list = []
    final_iteration_list = []
    # GPU
    # transform = np.genfromtxt("results/lidar_intensity_result/transformation_history_01_frame100.txt")
    # ell_history = np.genfromtxt("results/lidar_intensity_result/ell_history_01_frame100.txt")
    # indicator = np.genfromtxt("results/lidar_intensity_result/effective_points_01_frame100.txt")
    # CPU
    transform = np.genfromtxt("/home/cel/PERL/code/DockerFolder/outdoor_cvo/results/transformation_history_"+kitti_seq+"_"+str(gt_frame)+"_"+decay_rate+".txt")
    ell_history = np.genfromtxt("/home/cel/PERL/code/DockerFolder/outdoor_cvo/results/ell_history_"+kitti_seq+"_"+str(gt_frame)+"_"+decay_rate+".txt")
    indicator = np.genfromtxt("/home/cel/PERL/code/DockerFolder/outdoor_cvo/results/effective_points_"+kitti_seq+"_"+str(gt_frame)+"_"+decay_rate+".txt")
    inner_product = np.genfromtxt("/home/cel/PERL/code/DockerFolder/outdoor_cvo/results/inner_product_"+kitti_seq+"_"+str(gt_frame)+"_"+decay_rate+".txt")
    groundtruth = np.genfromtxt("/home/cel/PERL/code/DockerFolder/media/Samsung_T5/kitti/sequences/"+kitti_seq+"/groundtruth.txt")
    gt_now = groundtruth[gt_frame,:].reshape((3,4))
    gt_now = (np.vstack([gt_now, zero1]))
    gt_next = groundtruth[gt_frame+1,:].reshape((3,4))
    gt_next = np.vstack([gt_next, zero1])
    gt_curr = LA.inv(gt_now) @ gt_next

    # initial list to plot
    index_list = []
    t_error_list = []
    r_error_list = []
    ell_list = []
    indicator_list = []
    indicator_cdf_list = [0]
    inner_product_list = []
    index = 0
    
    # compute error
    num_row = transform.shape[0]
    
    for iteration in range(num_row-1):
        curr_tf = transform[iteration, :].reshape((3,4))
        curr_tf = np.vstack([curr_tf, zero1])
        error = linalg.inv(gt_curr) @ curr_tf
        r_err = rotationError(error)
        t_err = LA.norm(curr_tf[:3, 3] - gt_curr[:3, 3])

        # add to plot list
        index_list.append(index)
        t_error_list.append(t_err)
        r_error_list.append(r_err)
        ell_list.append(ell_history[iteration])
        indicator_list.append(indicator[iteration])
        indicator_cdf_list.append(indicator[iteration]+indicator_cdf_list[iteration])
        inner_product_list.append(inner_product[iteration])
        index += 1

    # to numpy array!
    index_list = np.array(index_list)
    t_error_list = np.array(t_error_list)
    r_error_list = np.array(r_error_list)
    ell_list = np.array(ell_list)
    indicator_list = np.array(indicator_list)
    indicator_cdf_list = np.array(indicator_cdf_list)
    inner_product_list = np.array(inner_product_list)

    # normalize...
    # t_error_list /= np.max(t_error_list)
    # r_error_list /= np.max(r_error_list)
    # ell_list /= np.max(ell_list)
    indicator_list /= np.max(indicator_list)
    indicator_cdf_list /= indicator_cdf_list[-1]
    inner_product_list /= np.max(inner_product_list)

    # plot indicator and error
    plt.figure(figsize=(1680/my_dpi, 1200/my_dpi), dpi=my_dpi)
    # plt.title("sequence %i, %s indicators with %.1f ratio to decay (normalized)" % (seq, indicator_a, float(indicator_b)/10))
    # fig, ax1 = plt.subplots(figsize=(1680/my_dpi, 800/my_dpi), dpi=my_dpi)

    ax = plt.subplot(211)
    ax.set_title('Kitti %s frame %i, decay rate %.1f, final error %.4f m' % (kitti_seq, gt_frame, float(decay_rate)/10, t_error_list[-1]))
    plt.ylabel('error, ell')
    plt.plot(index_list, ell_list, linestyle='dashed')
    plt.plot(index_list, t_error_list, linestyle='dotted')
    plt.plot(index_list, r_error_list, linestyle='dotted')
    plt.legend(["length scale", "translation error", "rotation error"], loc='upper right')
    ax = plt.gca()
    ax.set_xticklabels([])

    plt.subplot(212)
    plt.xlabel('iteration')
    plt.ylabel('indicator')
    # plt.plot(index_list, indicator_list, linestyle='solid', color='r')
    # plt.plot(index_list, indicator_cdf_list[1:], linestyle='dashed', color='g')
    plt.plot(index_list, inner_product_list, linestyle='solid', color='b')
    plt.legend(["normalized inner product"], loc='upper right')

    # plt.show()
    plt.savefig('%s/%s_%i_%.1f.png' % (folder_name, kitti_seq, gt_frame, float(decay_rate)/10), dpi=my_dpi)
    plt.close()


elif dataset == "lyft":
    for indicator_a, indicator_b in zip(a, b):
        folder_name = '/home/cel/Pictures/indicator/'+indicator_a+'_'+indicator_b
        Path(folder_name).mkdir(parents=True, exist_ok=True)
        final_t_error_list = []
        final_r_error_list = []
        final_iteration_list = []
        for seq in range(1):
            # transform = np.genfromtxt("results/lyft/transformation_history_"+str(seq)+"_"+indicator_a+"_"+indicator_b+"_ellmax_25.txt")
            # ell_history = np.genfromtxt("results/lyft/ell_history_"+str(seq)+"_"+indicator_a+"_"+indicator_b+"_ellmax_25.txt")
            # indicator = np.genfromtxt("results/lyft/effective_points_"+str(seq)+"_"+indicator_a+"_"+indicator_b+"_ellmax_25.txt")
            # transform = np.genfromtxt("results/lyft/transformation_history_"+str(seq)+"_"+indicator_a+"_"+indicator_b+"_nodropdetect_ellmax_25.txt")
            # ell_history = np.genfromtxt("results/lyft/ell_history_"+str(seq)+"_"+indicator_a+"_"+indicator_b+"_nodropdetect_ellmax_25.txt")
            # indicator = np.genfromtxt("results/lyft/effective_points_"+str(seq)+"_"+indicator_a+"_"+indicator_b+"_nodropdetect_ellmax_25.txt")
            transform = np.genfromtxt("results/lyft/transformation_history_"+str(seq)+"_expo.txt")
            ell_history = np.genfromtxt("results/lyft/ell_history_"+str(seq)+"_expo.txt")
            indicator = np.genfromtxt("results/lyft/effective_points_"+str(seq)+"_expo.txt")
            groundtruth = np.genfromtxt(seq_names[seq]+"/groundtruth.txt")
            gt_now = groundtruth[gt_frame,:].reshape((3,4))
            gt_now = (np.vstack([gt_now, zero1]))
            gt_next = groundtruth[gt_frame+1,:].reshape((3,4))
            gt_next = np.vstack([gt_next, zero1])
            gt_curr = LA.inv(gt_now) @ gt_next

            # initial list to plot
            index_list = []
            t_error_list = []
            r_error_list = []
            ell_list = []
            indicator_list = []
            indicator_cdf_list = [0]
            inner_product_list = []
            index = 0
            
            # compute error
            num_row = transform.shape[0]
            
            for iteration in range(num_row-1):
                curr_tf = transform[iteration, :].reshape((3,4))
                curr_tf = np.vstack([curr_tf, zero1])
                error = linalg.inv(gt_curr) @ curr_tf
                r_err = rotationError(error)
                t_err = LA.norm(curr_tf[:3, 3] - gt_curr[:3, 3])

                # add to plot list
                index_list.append(index)
                t_error_list.append(t_err)
                r_error_list.append(r_err)
                ell_list.append(ell_history[iteration])
                indicator_list.append(indicator[iteration])
                indicator_cdf_list.append(indicator[iteration]+indicator_cdf_list[iteration])
                inner_product_list.append(inner_product[iteration])
                index += 1

            # to numpy array!
            index_list = np.array(index_list)
            t_error_list = np.array(t_error_list)
            r_error_list = np.array(r_error_list)
            ell_list = np.array(ell_list)
            indicator_list = np.array(indicator_list)
            indicator_cdf_list = np.array(indicator_cdf_list)
            inner_product_list = np.array(inner_product_list)

            # normalize...
            # t_error_list /= np.max(t_error_list)
            # r_error_list /= np.max(r_error_list)
            # ell_list /= np.max(ell_list)
            indicator_list /= np.max(indicator_list)
            indicator_cdf_list /= indicator_cdf_list[-1]
            inner_product_list /= np.max(inner_product_list)

            # plot indicator and error
            plt.figure(figsize=(1680/my_dpi, 1200/my_dpi), dpi=my_dpi)
            # plt.title("sequence %i, %s indicators with %.1f ratio to decay (normalized)" % (seq, indicator_a, float(indicator_b)/10))
            # fig, ax1 = plt.subplots(figsize=(1680/my_dpi, 800/my_dpi), dpi=my_dpi)

            ax = plt.subplot(211)
            ax.set_title('Kitti %s frame %i, final error %.2f' % (kitti_seq, gt_frame, t_error_list[-1]))
            plt.ylabel('error, ell')
            plt.plot(index_list[:1000], ell_list[:1000], linestyle='dashed')
            plt.plot(index_list[:1000], t_error_list[:1000], linestyle='dotted')
            plt.plot(index_list[:1000], r_error_list[:1000], linestyle='dotted')
            plt.legend(["length scale", "translation error", "rotation error"], loc='upper right')
            ax = plt.gca()
            ax.set_xticklabels([])

            plt.subplot(212)
            plt.xlabel('iteration')
            plt.ylabel('indicator')
            # plt.plot(index_list[:1000], indicator_list[:1000], linestyle='solid', color='r')
            # plt.plot(index_list[:1000], indicator_cdf_list[1:301], linestyle='dashed', color='g')
            plt.plot(index_list[:1000], inner_product_list[:1000], linestyle='solid', color='b')
            plt.legend(["normalized indicator", "normalized inner product"], loc='upper right') # , "indicator CDF", 

            # plt.ylabel('error, ell')
            # plt.plot(index_list, ell_list, linestyle='dashed')
            # plt.plot(index_list, t_error_list, linestyle='dotted')
            # plt.plot(index_list, r_error_list, linestyle='dotted')
            # plt.legend(["length scale", "translation error", "rotation error"], loc='upper right')
            # ax = plt.gca()
            # ax.set_xticklabels([])

            # plt.subplot(212)
            # plt.xlabel('iteration')
            # plt.ylabel('indicator')
            # plt.plot(index_list, indicator_list, linestyle='solid', color='r')
            # plt.plot(index_list, indicator_cdf_list[1:], linestyle='dashed', color='g')
            # plt.plot(index_list, inner_product_list, linestyle='dotted', color='b')
            # plt.legend(["normalized indicator", "indicator CDF", "normalized inner product"], loc='upper right')

            plt.show()
            # plt.savefig('%s/%s.png' % (folder_name, str(seq)), dpi=my_dpi)
            plt.close()

        #     final_t_error_list.append(t_error_list[-1])
        #     final_r_error_list.append(r_error_list[-1])
        #     final_iteration_list.append(index_list[-1])
        # final_t_error_list = np.array(final_t_error_list)
        # final_r_error_list = np.array(final_r_error_list)
        # final_iteration_list = np.array(final_iteration_list)

        # print('success rate 0.1:', (np.sum(final_t_error_list < 0.1))/148*100)
        # print('average t-error:', np.mean(final_t_error_list))
        # print('average r-error:', np.mean(final_r_error_list))
        # print('average t-error for success:', np.mean(final_t_error_list[final_t_error_list < 0.1]))
        # print('average r-error for success:', np.mean(final_r_error_list[final_t_error_list < 0.1]))
        # print('average iteration:', np.mean(final_iteration_list))



