import numpy as np
import matplotlib.pyplot as plt

def main():
    rootpath = '/home/bigby/project/exp/tartanair_full_semantic/'
    angles_defined = ['12.5','25','37.5','50']
    outlier_defined = ['0.0','0.125','0.25','0.375','0.5']
    mode = "tartanair"
   
    prefixpath1 = 'tartanair_toy_exp_'

    prefixpath2 ='tartanair_toy_exp_'
   

    error_record = {}
    for outlier in outlier_defined:
        error_record[outlier] = {}
        error_record[outlier]['intencity'] = []
        error_record[outlier]['semantic'] = []
    num_exp = 10
    for angle in angles_defined:
        for outlier in outlier_defined:
            foldername_intencity = rootpath  + prefixpath1 + angle + '_' + outlier + '/'
            foldername_semantics = rootpath  + prefixpath2 + angle + '_' + outlier + '/'

            intencity_error_file = foldername_intencity + 'cvo_err_tartanair.txt'
            semantics_error_file = foldername_semantics + 'cvo_err_bunny_semantics.txt'
            intencity_error = np.loadtxt(intencity_error_file)
            semantics_error = np.loadtxt(semantics_error_file)
            average_intencity_error = np.average(intencity_error)
            average_semantics_error = np.average(semantics_error)
            error_record[outlier]['intencity'].append(average_intencity_error)
            error_record[outlier]['semantic'].append(average_semantics_error)
    angles = []
    for angle in angles_defined:
        angles.append(float(angle))

    color = ['b', 'g', 'r', 'c', 'm', 'y']
    index = 0
    for outlier in outlier_defined:
        plt.plot(angles, error_record[outlier]['intencity'], color=color[index],linestyle='-')
        index += 1
    
    index = 0
    for outlier in outlier_defined:
        plt.plot(angles, error_record[outlier]['semantic'], color=color[index],linestyle='--')
        index += 1
    
    outlier_legend_cvo = []
    outlier_legend_jrmpc = []
    for outlier in outlier_defined:
        outlier_legend_cvo.append(outlier + '_intencity')
        outlier_legend_jrmpc.append(outlier + '_semantic')
        
    plt.legend(outlier_legend_cvo + outlier_legend_jrmpc)
    plt.show()


            
            


if __name__ == '__main__':
    main()