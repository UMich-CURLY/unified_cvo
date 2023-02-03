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
    rootpath = '/home/bigby/project/exp/' 
    angles_defined = ['12.5','25','37.5','50']
    outlier_defined = ['0.0','0.125','0.25','0.375','0.5']
    #outlier_defined = ['0.25','0.375','0.5']
    mode = "tartanair"
    use_icp = True
    use_sem = True
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
                if use_icp:
                    error_entry['icp_error'].append(icp_error[i])
                    error_record[outlier]['icp_error'].append(icp_error[i])
                if use_sem:
                    error_entry['cvosem_error'].append(sem_error[i])
                    error_record[outlier]['cvosem_error'].append(sem_error[i])
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
    tablelegend(ax, ncol=3, bbox_to_anchor=(1, 1), 
            row_labels=['$i=1$', '$i=2$', '$i=3$','$i=4$','$i=5$'], 
            col_labels=['$j=1$', '$j=2$', '$j=3$'], 
            title_label='$f_{i,j}$')
    plt.show()

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
    
    
            
            


if __name__ == '__main__':
    main()