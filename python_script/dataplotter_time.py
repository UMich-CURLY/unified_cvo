import numpy as np
import matplotlib.pyplot as plt
import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle
import matplotlib.font_manager as fontmanager
from matplotlib import rc
import matplotlib

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def main():
    # plt.rcParams['text.usetex'] = True
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family' : "sans-serif"})
    plt.rc('text.latex', preamble=[r'\usepackage{amsmath}',r'\usepackage{amsfonts}'])
    
    rootpath = '/home/bigby/project/exp/' 
    angles_defined = ['12.5','25','37.5','50']
    outlier_defined = ['0.0','0.125','0.25','0.375','0.5']
    #outlier_defined = ['0.25','0.375','0.5']
    mode = "tartanair"
    use_icp = False

    use_sem = False
    if (mode == 'tartanair'):
        rootpath = '/home/bigby/project/exp/tartanair_full_semantic/'
        prefixpath = 'tartanair_toy_exp_'
        use_sem = True
    elif (mode == 'tartanair_semantics'):
        prefixpath = 'tartanair_toy_exp_semantic_'

    elif (mode == 'bunny'):
        prefixpath = 'toy_exp_'

    error_entry= {}
   

    error_entry['cvo_error'] = []
    error_entry['cvosem_error'] = []
    error_entry['jrmpc_error'] = []
    error_entry['jrmpc_error_large'] = []

    error_entry['icp_error'] = []
    error_entry['index'] = 0
    error_entry['exp'] = []
    
    semantic_total_time = 0
    error_record = {}
    for outlier in outlier_defined:
        error_record[outlier] = {}
        error_record[outlier]['cvo'] = []
        error_record[outlier]['cvosem'] = []
        error_record[outlier]['jrmpc'] = []
        error_record[outlier]['icp'] = []
        error_record[outlier]['cvo_error'] = []
        error_record[outlier]['cvosem_error'] = []
        error_record[outlier]['jrmpc_error'] = []
        error_record[outlier]['jrmpc_error_large'] = []
        error_record[outlier]['icp_error'] = []
        for angle in angles_defined:
            error_record[outlier][angle] = {}
            error_record[outlier][angle]['cvo'] = []
            error_record[outlier][angle]['jrmpc'] = []
            error_record[outlier][angle]['cvosem'] = []
            error_record[outlier][angle]['jrmpc_large'] = []
            error_record[outlier][angle]['icp'] = []
    num_exp = 10
    for outlier in outlier_defined:
        for angle in angles_defined:
            foldername = rootpath  + prefixpath + angle + '_' + outlier + '/'
            print(foldername)
            if (mode == "tartanair"):
                cvo_error_file = foldername + 'cvo_time_tartanair.txt'
                jrmpc_error_file = foldername + 'jrmpc_time.txt'
            elif (mode == "bunny"):
                cvo_error_file = foldername + 'cvo_time_bunny.txt'
                jrmpc_error_file = foldername + 'jrmpc_time.txt'
            if use_icp:
                icp_error_file = foldername + 'color_icp_err.txt'
                icp_error = np.loadtxt(icp_error_file)
            if use_sem: 
                sem_error_file = foldername + 'cvo_time_bunny_semantics.txt'
                sem_error = np.loadtxt(sem_error_file)
            jrmpc_large_error_file = foldername + 'jrmpc_time_large.txt'
            jrmpc_large_error = np.loadtxt(jrmpc_large_error_file)

            cvo_error = np.loadtxt(cvo_error_file)
            jrmpc_error = np.loadtxt(jrmpc_error_file)
            for i in range(cvo_error.shape[0]):
                error_entry['cvo_error'].append(cvo_error[i])
                error_entry['jrmpc_error'].append(jrmpc_error[i])
                error_entry['index'] +=1
                error_entry['exp'].append(error_entry['index'])
                error_record[outlier]['cvo_error'].append(cvo_error[i])
                error_record[outlier]['jrmpc_error'].append(jrmpc_error[i])
                error_record[outlier][angle]['cvo'].append(cvo_error[i])
                error_record[outlier][angle]['jrmpc'].append(jrmpc_error[i])
                error_entry['jrmpc_error_large'].append(jrmpc_large_error[i])
                error_record[outlier]['jrmpc_error_large'].append(jrmpc_large_error[i])
                error_record[outlier][angle]['jrmpc_large'].append(jrmpc_large_error[i])
                if use_icp:
                    error_entry['icp_error'].append(icp_error[i])
                    error_record[outlier]['icp_error'].append(icp_error[i])
                    error_record[outlier][angle]['icp'].append(icp_error[i])
                    
                if use_sem:
                    error_entry['cvosem_error'].append(sem_error[i])
                    error_record[outlier]['cvosem_error'].append(sem_error[i])
                    error_record[outlier][angle]['cvosem'].append(sem_error[i])
                    semantic_total_time += sem_error[i]
            average_cvo_error = np.average(cvo_error)
            average_jrmpc_error = np.average(jrmpc_error)
            if use_icp:
                average_icp_error = np.average(icp_error)
                error_record[outlier]['icp'].append(average_icp_error)
            if use_sem:
                average_sem_error = np.average(sem_error)
                error_record[outlier]['cvosem'].append(average_sem_error)
            error_record[outlier]['cvo'].append(average_cvo_error)
            error_record[outlier]['jrmpc'].append(average_jrmpc_error)
    angles = []
    for angle in angles_defined:
        angles.append(float(angle))
    print("semantic_total_time ",semantic_total_time)
    fontsize = 15
    # color = ['b', 'g', 'r', 'c', 'm', 'y']
    # index = 0
    # for outlier in outlier_defined:
    #     plt.plot(angles, error_record[outlier]['cvo'], color=color[index],linestyle='-',label= outlier + ' percent outlier_RKHS')
    #     index += 1
    
    # index = 0
    # for outlier in outlier_defined:
    #     plt.plot(angles, error_record[outlier]['jrmpc'], color=color[index],linestyle='--',label= outlier + ' percent outlier_JRMPC')
    #     index += 1
    # print(error_record['0.5']['icp'])
    # print(error_record['0.0']['icp'])
    # if use_icp:
    #     index = 0
    #     for outlier in outlier_defined:
    #         plt.plot(angles, error_record[outlier]['icp'], color=color[index],linestyle='-.',linewidth=2,label= outlier + ' percent outlier_Colored_ICP')
    #         index += 1
    # ax = plt.axes()
    outlier_legend_cvo = []
    outlier_legend_jrmpc = []
    outlier_legend_icp = []
    # for outlier in outlier_defined:
    #     outlier_legend_cvo.append(outlier + ' percent outlier_RKHS')
    #     outlier_legend_jrmpc.append(outlier + ' percent outlier_JRMPC')
    #     if use_icp:
    #         outlier_legend_icp.append(outlier + ' percent outlier_Colored_ICP')

    
    # plt.title("Angle Vs Error")
    # plt.xlabel("Angle / Degrees")

 

    # plt.ylabel(r'$\|\| T^{-1}*G-I \|\|_{F}$')
    # if use_icp:
    #     plt.legend()
    # else:
    #     plt.legend(ax,ncol=4,loc="center",bbox_to_anchor=(0.5, 1.1))
    # tablelegend(ax, ncol=3, bbox_to_anchor=(1, 1), 
    #         row_labels=['$i=1$', '$i=2$', '$i=3$','$i=4$','$i=5$'], 
    #         col_labels=['$j=1$', '$j=2$', '$j=3$'], 
    #         title_label='$f_{i,j}$')
    # plt.show()

    ## Plot second type of figure
    # for i in range(len(error_entry['exp'])):
    #     error_entry['exp'][i] = error_entry['exp'][i]/error_entry['exp'][len(error_entry['exp']) - 1]
    # fig = plt.figure(2)
    # plt.plot(error_entry['exp'], error_entry,'$i=4$','$i=5$'
    # plt.show()
    
    error_list_cvo = np.array(error_entry['cvo_error'])
    error_list_jrmpc = np.array(error_entry['jrmpc_error'])
    error_list_icp = np.array(error_entry['icp_error'])
    error_list_sem = np.array(error_entry['cvosem_error'])
    np.savetxt(rootpath  + 'cvo_error_list.txt', error_list_cvo,fmt='%.8f')
    np.savetxt(rootpath  + 'jrmpc_error_list.txt', error_list_jrmpc,fmt='%.8f')
    np.savetxt(rootpath  + 'icp_error_list.txt', error_list_icp,fmt='%.8f')
    np.savetxt(rootpath  + 'cvosem_error_list.txt', error_list_sem,fmt='%.8f')
    for outlier in outlier_defined:
        print(outlier)
        error_list_cvo = np.array(error_record[outlier]['cvo_error'])
        error_list_jrmpc = np.array(error_record[outlier]['jrmpc_error'])
        error_list_icp = np.array(error_record[outlier]['icp_error'])
        np.savetxt(rootpath  + 'cvo_error_list' + str(outlier ) + '.txt', error_list_cvo,fmt='%.8f')
        np.savetxt(rootpath  + 'jrmpc_error_list'+ str(outlier ) + '.txt', error_list_jrmpc,fmt='%.8f')
        np.savetxt(rootpath  + 'icp_error_list'+ str(outlier ) + '.txt', error_list_jrmpc,fmt='%.8f')
    
    # generate data for each of the three box plots in each subplot
    ls = ['0.0','12.0','25.0','37.5','50.0']
    if mode == "tartanair":
        # for angle in angles_defined:
          

            # rc('axes',**{'labelweight': 'bold'})
            # create a figure and axis
            fig, ax = plt.subplots(1, 5, figsize=(4.8, 3.6), sharey=True)
            i = 0
            w = 0.1
            # fig.suptitle(angle, fontsize=16)
            for outlier in outlier_defined:
                rc('font', **{'family': 'serif', 'serif': ['Computer Modern'],'weight' : 'bold'})
                rc('text', usetex=True)
                rc('font', weight='bold')
                # plot the first box plot in the first subplot
                medianprops = dict(color="red",linewidth=1.5)
                bp = ax[i].boxplot([np.array(error_record[outlier]['cvo']), np.array(error_record[outlier]['cvosem']) ,  np.array(error_record[outlier]['jrmpc']), np.array(error_record[outlier]['jrmpc_error_large'])],  widths = 1,
                vert=True, showfliers=False, patch_artist=True,medianprops=medianprops
                )
              
                # ax[i].set_title('Box Plot 1 of Error')
               
                

                ax[i].set_xticks([])
              
                if (i == 0):
                    ax[i].set_ylabel('Time (s)')
                
                    pass
                
                else:
                    #ax[i].set_yticks([])
                    
                    pass
                    # ax[i].set_yticks([])JRMPC gamma = 0.5
                ax[i].set_xlabel(ls[i],weight='bold')
                # ax[i].set_xticks([])
                #ax[i].xaxis.label.set_fontsize(18)
                # ax[i].yaxis.label.set_fontsize(18)
            
                colors = [[173/255, 216/255, 230/255],  [159/255, 43/255, 104/255], [144/255, 238/255, 144/255],[181/255, 101/255, 29/255]]
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                if (i == 0):  
                    ax[i].legend(handles=bp['boxes'], labels=['Color RKHS-BA','Semantics RKHS-BA','JRMPC gamma = 0.1','JRMPC gamma = 0.5'],  bbox_to_anchor=(5.2, 1.31), ncol=2,prop={'size': 12.5,'weight': 'bold'})
                    # ax[i].xaxis.set_label_coords(2.5, - 0.07)
                    # ax[i].set_xlabel("Outlier Ratio %", fontsize=18)
            
                ax[i].tick_params(axis='both', which='major', labelsize=fontsize)
                ax[i].xaxis.label.set_size(fontsize)
                ax[i].yaxis.label.set_size(fontsize)
                ax[i].set_yscale("log")
                i = i+1
                
            handles, labels = ax[0].get_legend_handles_labels()
            # ax[0].set_yticklabels(ax[0].get_yticks(), weight='bold')

        
            # font = {'family': 'Computer Modern',
            # 'weight': 'bold',
            # 'size': 18,
            # }
            # fig.text(0.5, 0.01, r' \boldmath $Outlier\;Ratio\;(\%)$', fontsize=18,ha ='center')
            fig.text(0.52, 0.01, r'\boldmath Outlier Ratio (\%)', ha ='center',fontsize = 19)
            # fig.legend(*ax[0].get_legend_handles_labels(),loc='center',  bbox_to_anchor=(0.5, 0.91), ncol=3)
            # fig.legend(['RKHS_Intencity', 'JRMPC', 'Colored_ICP'],loc='center',  bbox_to_anchor=(0.5, 0.91), ncol=3)
            plt.subplots_adjust(
top=0.81,
bottom=0.13,
left=0.135,
right=0.965,
hspace=0.2,
wspace=0.0)
            fig.canvas.draw()
            plt.show()
            # fig.legend(loc='center', bbox_to_anchor=(0.5, -0.1), ncol=3)     
    else:
        # for angle in angles_defined:
            rc('font', **{'family': 'serif', 'serif': ['Computer Modern'],'weight' : 'bold'})
            rc('text', usetex=True)
            # create a figure and axis
            fig, ax = plt.subplots(1, 5, figsize=(4.8, 3.6), sharey=True)
            i = 0
            # fig.suptitle(angle, fontsize=16)
            medianprops = dict(color="red",linewidth=1.5)
            for outlier in outlier_defined:
                # plot the first box plot in the first subplot
               
                bp = ax[i].boxplot([np.array(error_record[outlier]['cvo']),  np.array(error_record[outlier]['jrmpc']),np.array(error_record[outlier]['jrmpc_error_large'])], 
                vert=True, showfliers=False, patch_artist=True,widths = 1,medianprops=medianprops
                )
              
                # ax[i].set_title('Box Plot 1 of Error')
               
                

                ax[i].set_xticks([])
              
                if (i == 0):
                    ax[i].set_ylabel('Time (s)')
                
                else:
                    #ax[i].set_yticks([])
                    
                    pass
                    # ax[i].set_yticks([])
                ax[i].set_xlabel(ls[i],weight='bold')
                # ax[i].set_xticks([])
                #ax[i].xaxis.label.set_fontsize(18)
                # ax[i].yaxis.label.set_fontsize(18)

                colors = [[173/255, 216/255, 230/255], [144/255, 238/255, 144/255],[181/255, 101/255, 29/255]]
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                if (i == 0):  
                    ax[i].legend(handles=bp['boxes'], labels=['RKHS-BA', 'JRMPC gamma = 0.1','JRMPC gamma = 0.5'],  bbox_to_anchor=(5.29, 1.31), ncol=2,prop={'size': 12.5})
                    # ax[i].xaxis.set_label_coords(2.5, - 0.07)
                    # ax[i].set_xlabel("Outlier Ratio %", fontsize=18)
                ax[i].tick_params(axis='both', which='major', labelsize=fontsize)
                ax[i].xaxis.label.set_size(fontsize)
                ax[i].yaxis.label.set_size(fontsize)
                ax[i].set_yscale("log")
                i = i+1
                
            handles, labels = ax[0].get_legend_handles_labels()
            
        
            # font = {'family': 'Computer Modern',
            # 'weight': 'bold',
            # 'size': 18,
            # }
            # fig.text(0.5, 0.01, r'$Outlier\;Ratio\;(\%)$', fontsize=18,ha ='center')
            fig.text(0.5, 0.01, 'Outlier Ratio (\%)', ha ='center',fontsize = fontsize)
            # fig.legend(*ax[0].get_legend_handles_labels(),loc='center',  bbox_to_anchor=(0.5, 0.91), ncol=3)
            # fig.legend(['RKHS_Intencity', 'JRMPC', 'Colored_ICP'],loc='center',  bbox_to_anchor=(0.5, 0.91), ncol=3)
            plt.subplots_adjust(
            top=0.81,
            bottom=0.13,
            left=0.135,
            right=0.965,
            hspace=0.2,
            wspace=0.0)
            fig.canvas.draw()
            plt.show()
            # fig.legend(loc='center',       
        

            
            


if __name__ == '__main__':
    main()