import open3d as o3d
import numpy as np
import copy
if __name__ == "__main__":
    # load the mesh
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
    outlier_defined = ['0.0','0.1','0.2','0.3','0.4','0.5']
    angles_defined = ['10','20','30','40','50']
    exp = 0
    pointcloudFolder = rootpath + prefixpath +  str(angles_defined[4]) + '_' + str(outlier_defined[5]) + '/' + str(exp) + '/'
    colors = [[1,0,0],[0,1,0],[0,0,1],[1,1,0]]
    print(pointcloudFolder)
    
    # read the pointcloud use point cloud folder with open3d\
    pc = []
    for i in range(4):
        filename = pointcloudFolder + str(i) + 'normal.pcd'
        newpc = o3d.io.read_point_cloud(filename)
        newpc.paint_uniform_color(colors[i])
        pc.append(newpc)
    o3d.visualization.draw_geometries([pc[0],pc[1],pc[2],pc[3]])
    

    # transform to cvo result 
    # load cvo pose \
    cvotransformation = pointcloudFolder + 'rkhs_results.txt'
    jrmpctransformation = pointcloudFolder + 'jrmpc.txt'
    def transformed_pointlcloud(filename,pts):
        # load transformation
        tr = np.loadtxt(filename)
        pt_t = []
        for i in range(4):
            T = tr[i,:].reshape(4,4)
            pt = copy.deepcopy(pts[i]).transform(T)
            pt_t.append(pt)
        return pt_t

        

    pt_t = transformed_pointlcloud(cvotransformation,pc)
    o3d.visualization.draw_geometries([pt_t[0],pt_t[1],pt_t[2],pt_t[3]])

    # # transform to jrmpc result
    # pt_t = transformed_pointlcloud(jrmpctransformation,pc)
    # o3d.visualization.draw_geometries([pt_t[0],pt_t[1],pt_t[2],pt_t[3]])
