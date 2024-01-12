import numpy as np
from matplotlib.pyplot as plt


angles_defined = [90, 180]
outlier_defined = [0.0, 0.125, 0.25, 0.375, 0.5]
crop_defined = [0.0, 0.125, 0.25, 0.375, 0.5]

def read_data(method_list,
              angles_defined,
              crop_defined,
              outlier_defined,
              folder_suffix):
    error_record = {}

    for angle in angles_defined:
        error_record[angle] = {}
        for crop in crop_defined: 
            error_record[angle][crop] = {}           
            for outlier in outlier_defined:
                folder = "result_global_"+angle+"_"+outlier+"_"+crop+"_"+folder_suffix
                err = np.genfromtxt(folder+"/fro_err_"+method+".txt")
                error_record[angle][crop][outlier] = err

    return error_record
    
def make_boxplot(method_list,
                 angles_defined,
                 crop_defined,
                 outlier_defined,
                 error_record):
    bplot1 = ax1.boxplot(all_data,
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=labels)  # will be used to label x-ticks
    ax1.set_title('Rectangular box plot')


    error_record = read_data()
    

    for angle in angles_defined:
        rc('font', **{'family': 'serif', 'serif': ['Computer Modern'],'weight' : 'bold'})
        rc('text', usetex=True)
        
        for crop in crop_defined:
            # each subplot
            fig, ax = plt.subplots(1, 5, figsize=(4.8, 3.6), sharey=True)            
            for outlier in outlier_defined:
                # each panel inside the subplot
                i = 0
                # fig.suptitle(angle, fontsize=16)
                
                # plot the first box plot in the first subplot
                bp = ax[i].boxplot([np.array(error_record[angle][crop][outlier][method]) for method in method_list ],
                                   vert=True, showfliers=False, patch_artist=True, widths = 0.5
                                   )
              
                # ax[i].set_title('Box Plot 1 of Error')
                ax[i].set_xticks([])
            
                if (i == 0):
                    ax[i].set_ylabel(r'$\lVert T^{-1}G-I \rVert_{F} $')
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
                    ax[i].legend(handles=bp['boxes'], labels=method_list,  bbox_to_anchor=(5.29, 1.31), ncol=2,prop={'size': 12.5})
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
                # fig.legend(loc='center', bbox_t

    




if __name__ == '__main__':

    method_list = ['cvo', 'fgr', 'ransac' ]
    angles_defined = [90, 180]
    outlier_defined = [0.0, 0.125, 0.25, 0.375, 0.5]
    crop_defined = [0.0, 0.125, 0.25, 0.375, 0.5]

    error_record = read_data(method_list,
                             angles_defined,
                             crop_defined,
                             outlier_defined)
    
    make_boxplot(method_list,
                 angles_defined,
                 crop_defined,
                 outlier_defined,
                 error_record)
    
