# Python 3 code to rename multiple
# files in a directory or folder
 
# importing os module
import os
import shutil
# Function to rename multiple files
def main():
    rootpath = '/home/bigby/project/unified_cvo/'
    angles_defined = ['12.5','25','37.5','50']
    outlier_defined = ['0.0','0.125','0.25','0.375','0.5']
    prefixpath = 'tartanair_toy_exp_'
    num_exp = 40
 

    for angle in angles_defined:
        for outlier in outlier_defined:
            for i in range(num_exp):
                foldername = rootpath  + prefixpath + angle + '_' + outlier + '/' + str(i) + '/'
                print(foldername)
                cvo_error_file = foldername + 'error_rksh_results.txt'
                cvo_pose_file = foldername + 'rkhs_results.txt'
                cvo_error_file_dst = foldername + 'error_rkhs_intencity_results.txt'
                cvo_pose_file_dst = foldername + 'rkhs_intencity_result.txt'
                shutil.copy(cvo_error_file, cvo_error_file_dst)
                shutil.copy(cvo_pose_file, cvo_pose_file_dst)
            
    angles = []
    for angle in angles_defined:
        angles.append(float(angle))
# Driver Code
if __name__ == '__main__':
     
    # Calling main() function
    main()