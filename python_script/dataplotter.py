import numpy as np
import matplotlib.pyplot as plt
import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle
def tablelegend(ax, col_labels=None, row_labels=None, title_label="", *args, **kwargs):
    """
    Place a table legend on the axes.
    
    Creates a legend where the labels are not directly placed with the artists, 
    but are used as row and column headers, looking like this:
    
    title_label   | col_labels[1] | col_labels[2] | col_labels[3]
    -------------------------------------------------------------
    row_labels[1] |
    row_labels[2] |              <artists go there>
    row_labels[3] |
    
    
    Parameters
    ----------
    
    ax : `matplotlib.axes.Axes`
        The artist that contains the legend table, i.e. current axes instant.
        
    col_labels : list of str, optional
        A list of labels to be used as column headers in the legend table.
        `len(col_labels)` needs to match `ncol`.
        
    row_labels : list of str, optional
        A list of labels to be used as row headers in the legend table.
        `len(row_labels)` needs to match `len(handles) // ncol`.
        
    title_label : str, optional
        Label for the top left corner in the legend table.
        
    ncol : int
        Number of columns.
        

    Other Parameters
    ----------------
    
    Refer to `matplotlib.legend.Legend` for other parameters.
    
    """
    #################### same as `matplotlib.axes.Axes.legend` #####################
    handles, labels, extra_args, kwargs = mlegend._parse_legend_args([ax], *args, **kwargs)
    if len(extra_args):
        raise TypeError('legend only accepts two non-keyword arguments')
    
    if col_labels is None and row_labels is None:
        ax.legend_ = mlegend.Legend(ax, handles, labels, **kwargs)
        ax.legend_._remove_method = ax._remove_legend
        return ax.legend_
    #################### modifications for table legend ############################
    else:
        ncol = kwargs.pop('ncol')
        handletextpad = kwargs.pop('handletextpad', 0 if col_labels is None else -2)
        title_label = [title_label]
        
        # blank rectangle handle
        extra = [Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)]
        
        # empty label
        empty = [""]
        
        # number of rows infered from number of handles and desired number of columns
        nrow = len(handles) // ncol
        
        # organise the list of handles and labels for table construction
        if col_labels is None:
            assert nrow == len(row_labels), "nrow = len(handles) // ncol = %s, but should be equal to len(row_labels) = %s." % (nrow, len(row_labels))
            leg_handles = extra * nrow
            leg_labels  = row_labels
        elif row_labels is None:
            assert ncol == len(col_labels), "ncol = %s, but should be equal to len(col_labels) = %s." % (ncol, len(col_labels))
            leg_handles = []
            leg_labels  = []
        else:
            assert nrow == len(row_labels), "nrow = len(handles) // ncol = %s, but should be equal to len(row_labels) = %s." % (nrow, len(row_labels))
            assert ncol == len(col_labels), "ncol = %s, but should be equal to len(col_labels) = %s." % (ncol, len(col_labels))
            leg_handles = extra + extra * nrow
            leg_labels  = title_label + row_labels
        for col in range(ncol):
            if col_labels is not None:
                leg_handles += extra
                leg_labels  += [col_labels[col]]
            leg_handles += handles[col*nrow:(col+1)*nrow]
            leg_labels  += empty * nrow
        
        # Create legend
        ax.legend_ = mlegend.Legend(ax, leg_handles, leg_labels, ncol=ncol+int(row_labels is not None), handletextpad=handletextpad, **kwargs)
        ax.legend_._remove_method = ax._remove_legend
        return ax.legend_
def main():
    # plt.rcParams['text.usetex'] = True
    plt.rc('text', usetex=True)
    plt.rc('font', **{'family' : "sans-serif"})
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    
    rootpath = '/home/bigby/project/exp/' 
    angles_defined = ['12.5','25','37.5','50']
    outlier_defined = ['0.0','0.125','0.25','0.375','0.5']
    #outlier_defined = ['0.25','0.375','0.5']
    mode = "bunny"
    use_icp = False
    use_sem = False
    if (mode == 'tartanair'):
        rootpath = '/home/bigby/project/exp/tartanair_full_semantic/'
        prefixpath = 'tartanair_toy_exp_'

    elif (mode == 'tartanair_semantics'):
        prefixpath = 'tartanair_toy_exp_semantic_'

    elif (mode == 'bunny'):
        prefixpath = 'toy_exp_'

    error_entry= {}
   

    error_entry['cvo_error'] = []
    error_entry['cvosem_error'] = []
    error_entry['jrmpc_error'] = []
    error_entry['icp_error'] = []
    error_entry['index'] = 0
    error_entry['exp'] = []
    

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
        error_record[outlier]['icp_error'] = []
        for angle in angles_defined:
            error_record[outlier][angle] = {}
            error_record[outlier][angle]['cvo'] = []
            error_record[outlier][angle]['jrmpc'] = []
            error_record[outlier][angle]['cvosem'] = []
            error_record[outlier][angle]['icp'] = []
    num_exp = 10
    for outlier in outlier_defined:
        for angle in angles_defined:
            foldername = rootpath  + prefixpath + angle + '_' + outlier + '/'
            print(foldername)
            if (mode == "tartanair"):
                cvo_error_file = foldername + 'cvo_err_tartanair.txt'
                jrmpc_error_file = foldername + 'jrmpc_error.txt'
            elif (mode == "bunny"):
                cvo_error_file = foldername + 'cvo_err_bunny.txt'
                jrmpc_error_file = foldername + 'jrmpc_error.txt'
            if use_icp:
                icp_error_file = foldername + 'color_icp_err.txt'
                icp_error = np.loadtxt(icp_error_file)
            if use_sem:
                sem_error_file = foldername + 'cvo_err_bunny_semantics.txt'
                sem_error = np.loadtxt(sem_error_file)


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
                if use_icp:
                    error_entry['icp_error'].append(icp_error[i])
                    error_record[outlier]['icp_error'].append(icp_error[i])
                    error_record[outlier][angle]['icp'].append(icp_error[i])
                if use_sem:
                    error_entry['cvosem_error'].append(sem_error[i])
                    error_record[outlier]['cvosem_error'].append(sem_error[i])
                    error_record[outlier][angle]['cvosem'].append(sem_error[i])
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

    color = ['b', 'g', 'r', 'c', 'm', 'y']
    index = 0
    for outlier in outlier_defined:
        plt.plot(angles, error_record[outlier]['cvo'], color=color[index],linestyle='-',label= outlier + ' percent outlier_RKHS')
        index += 1
    
    index = 0
    for outlier in outlier_defined:
        plt.plot(angles, error_record[outlier]['jrmpc'], color=color[index],linestyle='--',label= outlier + ' percent outlier_JRMPC')
        index += 1
    print(error_record['0.5']['icp'])
    print(error_record['0.0']['icp'])
    if use_icp:
        index = 0
        for outlier in outlier_defined:
            plt.plot(angles, error_record[outlier]['icp'], color=color[index],linestyle='-.',linewidth=2,label= outlier + ' percent outlier_Colored_ICP')
            index += 1
    ax = plt.axes()
    outlier_legend_cvo = []
    outlier_legend_jrmpc = []
    outlier_legend_icp = []
    # for outlier in outlier_defined:
    #     outlier_legend_cvo.append(outlier + ' percent outlier_RKHS')
    #     outlier_legend_jrmpc.append(outlier + ' percent outlier_JRMPC')
    #     if use_icp:
    #         outlier_legend_icp.append(outlier + ' percent outlier_Colored_ICP')

    
    # plt.title("Angle Vs Error")
    plt.xlabel("Angle / Degrees")
    plt.ylabel(r'$\|\| T^{-1}*G-I \|\|_{F}$')
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
   
    if use_sem:
        for angle in angles_defined:
            # create a figure and axis
            fig, ax = plt.subplots(1, 5, figsize=(6.4, 4.8), sharey=True)
            i = 0
            # fig.suptitle(angle, fontsize=16)
            for outlier in outlier_defined:
                # plot the first box plot in the first subplot
                # print(np.array(error_record[outlier][angle]['cvo']).shape)
                bp = ax[i].boxplot([np.array(error_record[outlier][angle]['cvo']), np.array(error_record[outlier][angle]['cvosem']) ,  np.array(error_record[outlier][angle]['jrmpc'])], 
                vert=True, showfliers=False, patch_artist=True,
                labels=['RKHS_Intencity', 'JRMPC', 'Colored_ICP'])
                ax[i].set_xlabel(float(outlier)*100)
                ax[i].set_xticks([])
                # ax[i].set_title('Box Plot 1 of Error')
                ax[i].xaxis.label.set_fontsize(15)
                ax[i].yaxis.label.set_fontsize(15)
                ax[i].xaxis.set_tick_params(labelsize=12)
                if (i == 0):
                    ax[i].set_ylabel(r'$\|\| T^{-1}*G-I \|\|_{F}$')
                
                else:
                    # ax[i].set_yticks([])
                    
                    pass

                colors = [[173/255, 216/255, 230/255], [255/255, 255/255, 224/255], [144/255, 238/255, 144/255]]
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                if (i == 0):  
                    ax[i].legend(handles=bp['boxes'], labels=['RKHS_Intencity', 'RKHS_Semantics', 'JRMPC'],  bbox_to_anchor=(4.8, 1.105), ncol=3)
                    ax[i].xaxis.set_label_coords(2.5, - 0.07)
                    ax[i].set_xlabel("Outlier Ratio %", fontsize=18)
            
                i = i+1
            handles, labels = ax[0].get_legend_handles_labels()
            
            # fig.legend(*ax[0].get_legend_handles_labels(),loc='center',  bbox_to_anchor=(0.5, 0.91), ncol=3)
            # fig.legend(['RKHS_Intencity', 'JRMPC', 'Colored_ICP'],loc='center',  bbox_to_anchor=(0.5, 0.91), ncol=3)
            plt.subplots_adjust(wspace=0)
            plt.show()
            # fig.legend(loc='center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    else:
        for angle in angles_defined:
            # create a figure and axis
            fig, ax = plt.subplots(1, 5, figsize=(6.4, 4.8), sharey=True)
            i = 0
            # fig.suptitle(angle, fontsize=16)
            for outlier in outlier_defined:
                # plot the first box plot in the first subplot
                print(np.array(error_record[outlier][angle]['cvo']).shape)
                print(np.array(error_record[outlier][angle]['jrmpc']).shape)
                bp = ax[i].boxplot([np.array(error_record[outlier][angle]['cvo']),  np.array(error_record[outlier][angle]['jrmpc'])], 
                vert=True, showfliers=False, patch_artist=True,
                labels=['RKHS_Intencity', 'JRMPC'])
              
                # ax[i].set_title('Box Plot 1 of Error')
               
                

                ax[i].set_xticks([])
                if (i == 0):
                    ax[i].set_ylabel(r'$\lVert T^{-1}G-I \rVert_{F} $')
                
                else:
                    # ax[i].set_yticks([])
                    
                    pass
                    # ax[i].set_yticks([])
                ax[i].set_xlabel(str(float(outlier)*100))
                ax[i].xaxis.label.set_fontsize(18)
                ax[i].yaxis.label.set_fontsize(18)

                colors = [[173/255, 216/255, 230/255], [144/255, 238/255, 144/255]]
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                if (i == 0):  
                    ax[i].legend(handles=bp['boxes'], labels=['RKHS', 'JRMPC'],  bbox_to_anchor=(4.45, 1.17), ncol=3,prop={'size': 18})
                    # ax[i].xaxis.set_label_coords(2.5, - 0.07)
                    # ax[i].set_xlabel("Outlier Ratio %", fontsize=18)
                ax[i].tick_params(axis='both', which='major', labelsize=18)
                i = i+1
                
            handles, labels = ax[0].get_legend_handles_labels()
            font = {'family': "serif",
            'weight': 'normal',
            'size': 18,
            }

            fig.text(0.5, 0.02, 'Outlier Ratio\%', fontsize=18,ha ='center')
            # fig.legend(*ax[0].get_legend_handles_labels(),loc='center',  bbox_to_anchor=(0.5, 0.91), ncol=3)
            # fig.legend(['RKHS_Intencity', 'JRMPC', 'Colored_ICP'],loc='center',  bbox_to_anchor=(0.5, 0.91), ncol=3)
            plt.subplots_adjust(wspace=0)
            plt.show()
            # fig.legend(loc='center', bbox_to_anchor=(0.5, -0.1), ncol=3)           
        

            
            


if __name__ == '__main__':
    main()