import open3d as o3d
import numpy as np
import copy    
def save_trans(trans, filename):
    # save transformation
    trans_format = []
    for i in range(4):
        trans_format.append(trans[i].reshape(16))
    np.savetxt(filename,np.array(trans_format).reshape(4,16), fmt='%.8f')
def load_transform(transform_path):
    # load transformation
    tr = np.loadtxt(transform_path)
    trans = []
    for i in range(4):
        T = tr[i,:].reshape(4,4)
        trans.append(T)
    return trans
def calculate_error(pred, gt,filename):
    errors = [0]
    totalerror = 0
    for i in range(1,4):
        current_pred = pred[i]
        current_gt = gt[i]
        error = np.linalg.norm(current_pred-current_gt, 'fro')
        errors.append(error)
        totalerror += error
    print(filename)
    print(str(errors[0]) +' '+ str(errors[1]) +' '+ str(errors[2]))
    np.savetxt(filename,np.array(errors).reshape(4,1), fmt='%.8f')
    return totalerror
    
    
    
def load_exp(filename):
    # load transformation
    tr = np.loadtxt(filename)
    trans = []
    for i in range(4):
        T = tr[i,:].reshape(4,4)
        trans.append(T)
    return trans
def transformed_pointlcloud(trans,pts):     
    result_pt = []
    for i in range(4):
        T = trans[i]
        pt = copy.deepcopy(pts[i]).transform(T)
        result_pt.append(pt)
    return result_pt
def main():
    rootpath = '/home/bigby/project/unified_cvo/'
    angles_defined = ['12.5','25','37.5','50']
    outlier_defined = ['0.0','0.125','0.25','0.375','0.5']
    prefixpath = 'tartanair_toy_exp_'
    num_exp = 40
    exp = 0
    pointcloudFolder = rootpath + prefixpath +  str(angles_defined[0]) + '_' + str(outlier_defined[0]) + '/' + str(exp) + '/'
    max_iter = 100
    pc = []
    radius = 1
    for i in range(4):
        filename = pointcloudFolder + str(i) + 'normal_color.pcd'
        newpc = o3d.io.read_point_cloud(filename)
        # newpc.paint_uniform_color(colors[i])
        # calculate normal
        newpc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=40))
        pc.append(newpc)
    o3d.visualization.draw_geometries([pc[0],pc[1],pc[2],pc[3]])
    result_transformation = [np.identity(4)]
    
    for i in range(1,4):
        current_transformation = np.identity(4)
        result_icp = o3d.pipelines.registration.registration_colored_icp(
        pc[i],pc[0] , radius, current_transformation,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                          relative_rmse=1e-6,
                                                          max_iteration=max_iter))
        current_transformation = result_icp.transformation
        result_transformation.append(current_transformation)
    print(result_transformation)
    pt_t = transformed_pointlcloud(result_transformation,pc)
    o3d.visualization.draw_geometries([pt_t[0],pt_t[1],pt_t[2],pt_t[3]])
    # load gt transformation
    gtpath = pointcloudFolder + 'gt_poses.txt'
    gt_t = load_transform(gtpath)
    # transfer gt
    for i in range(4):
        gt_t[i] = np.linalg.inv(gt_t[i])
    totalerror = calculate_error(result_transformation,gt_t,pointcloudFolder + 'error_color_icp.txt')
    # save icp result 
    save_trans(result_transformation,pointcloudFolder + 'color_icp.txt')
if __name__ == '__main__':  
    main()

    # for angle in angles_defined:
    #     for outlier in outlier_defined:
    #         for i in range(num_exp):
    #             foldername = rootpath  + prefixpath + angle + '_' + outlier + '/' + str(i) + '/'
    #             print(foldername)
    #             cvo_error_file = foldername + 'error_rkhs_results.txt'
    #             cvo_pose_file = foldername + 'rkhs_results.txt'
    #             cvo_error_file_dst = foldername + 'error_rkhs_intencity_results.txt'
    #             cvo_pose_file_dst = foldername + 'rkhs_intencity_result.txt'
    #             shutil.copy(cvo_error_file, cvo_error_file_dst)
    #             shutil.copy(cvo_pose_file, cvo_pose_file_dst)