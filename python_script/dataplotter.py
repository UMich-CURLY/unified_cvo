import numpy as np
import matplotlib.pyplot as plt

def main():
    rootpath = '/home/bigby/project/exp/'
    angles_defined = ['10','20','30','40','50']
    outlier_defined = ['0.0','0.1','0.2','0.3','0.4','0.5']
    mode = "bunny"
    if (mode == 'tartanair'):
        prefixpath = 'tartanair_toy_exp_'

    elif (mode == 'tartanair_semantics'):
        prefixpath = 'tartanair_toy_exp_semantic_'

    elif (mode == 'bunny'):
        prefixpath = 'toy_exp_'

    error_record = {}
    for outlier in outlier_defined:
        error_record[outlier] = {}
        error_record[outlier]['cvo'] = []
        error_record[outlier]['jrmpc'] = []
    num_exp = 10
    for angle in angles_defined:
        for outlier in outlier_defined:
            foldername = rootpath  + prefixpath + angle + '_' + outlier + '/'
            print(foldername)
            if (mode == "tartanair"):
                cvo_error_file = foldername + 'cvo_err_tartanair.txt'
                jrmpc_error_file = foldername + 'jrmpc_error.txt'
            elif (mode == "bunny"):
                cvo_error_file = foldername + 'cvo_err_bunny.txt'
                jrmpc_error_file = foldername + 'jrmpc_error.txt'
            cvo_error = np.loadtxt(cvo_error_file)
            jrmpc_error = np.loadtxt(jrmpc_error_file)
            average_cvo_error = np.average(cvo_error)
            average_jrmpc_error = np.average(jrmpc_error)
            error_record[outlier]['cvo'].append(average_cvo_error)
            error_record[outlier]['jrmpc'].append(average_jrmpc_error)
    angles = []
    for angle in angles_defined:
        angles.append(float(angle))

    color = ['b', 'g', 'r', 'c', 'm', 'y']
    index = 0
    for outlier in outlier_defined:
        plt.plot(angles, error_record[outlier]['cvo'], color=color[index],linestyle='-')
        index += 1
    
    index = 0
    for outlier in outlier_defined:
        plt.plot(angles, error_record[outlier]['jrmpc'], color=color[index],linestyle='--')
        index += 1
    
    outlier_legend_cvo = []
    outlier_legend_jrmpc = []
    for outlier in outlier_defined:
        outlier_legend_cvo.append(outlier + '_cvo')
        outlier_legend_jrmpc.append(outlier + '_jrmpc')
        
    plt.legend(outlier_legend_cvo + outlier_legend_jrmpc)
    plt.show()


            
            


if __name__ == '__main__':
    main()