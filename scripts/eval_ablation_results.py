import numpy as np
import matplotlib.pyplot as plt 

def box_plot(ratio_list, x):

    fontsize = 10
    fig, ax = plt.subplots(1, len(ratio_list), figsize=(4.8, 3.6), sharey=True)
    colors = [[173/255, 216/255, 230/255],
              [144/255, 238/255, 144/255],
              [181/255, 101/255, 29/255]]
    i = 0
    for ratio in ratio_list:
        
        bp = ax[i].boxplot(x[i],
                        vert=True, showfliers=False, patch_artist=True, widths = 0.5)
        
        ax[i].set_xticks([])
        ax[i].set_xlabel(ratio)
        for patch in bp['boxes']:
            patch.set_facecolor(colors[0])

        if i == 0:
            ax[i].set_ylabel('Pose Error')
            #ax[i].legend(handles=bp['boxes'], labels=['Semantic RKHS-BA Error'],
            #             bbox_to_anchor=(4.05, 1.31), ncol=2,
            #             prop={'size': 12.5})
                    
        #ax[i].tick_params(axis='both', which='major', labelsize=fontsize)
        ax[i].xaxis.label.set_size(fontsize)
        ax[i].yaxis.label.set_size(fontsize)
        #ax[i].set_yscale("log")
        fig.text(0.55, 0.01, 'Semantic Noise Sigma', ha ='center',fontsize = fontsize)
        plt.subplots_adjust(
            top=0.81,
            bottom=0.13,
            left=0.135,
            right=0.965,
            hspace=0.2,
            wspace=0.0)
        #fig.canvas.draw()
        #fig.savefig("angle_"+str(angle)+"_"+str(crop)+"_"+str(outlier)+".png", dpi=300)
        
        i+=1
    
    plt.show()    


def eval_seq_mean_err(prefix, difficulty, seqs, ratio_list, date):
    errs = []
    for ratio in ratio_list:
        print("{} noise sigma: {}".format(prefix, ratio))
        log_err_all = []
        fro_err_all = []
        for seq in seqs:
            folder =  "{}/{}/tartan_ablation_{}_{}_{}".format(prefix, ratio, difficulty, seq, date)
            log_err = np.genfromtxt(folder + "/log_err.txt")
            log_err_all.append(log_err)

            fro_err = np.genfromtxt(folder + "/fro_err.txt")
            fro_err_all.append(fro_err)
        log_err_all = np.concatenate(log_err_all)
        print("log_err_all shape is {}, mean is {}, std is {}".format( log_err_all.shape, np.mean(log_err_all), np.std(log_err_all)))
        fro_err_all = np.concatenate(fro_err_all)
        print("fro_err_all shape is {}, mean is {}, std is {}".format( fro_err_all.shape, np.mean(fro_err_all), np.std(fro_err_all)))
        errs.append(fro_err_all)
    box_plot(ratio_list, errs)
            


if __name__ == "__main__":
    #   (eval_seq_mean_err('semantic', 'Easy', ['abandonedfactory', 'gascola','seasonsforest', 'seasonsforest_winter', 'soulcity' ], ['0.0',  '0.1',   '0.25', '0.5',  '1',  '2',  '4' , '8'], 'dec15'))
    (eval_seq_mean_err('semantic_ablation/semantic', 'Easy', ['abandonedfactory', 'gascola','seasonsforest', 'seasonsforest_winter', 'soulcity' ], ['0.0',   '0.25', '0.5',  '1',  '2'], 'dec15'))
