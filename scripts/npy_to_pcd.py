import open3d as o3d
import numpy as np
import sys, os
from tqdm import tqdm
def npy_to_pcd(in_fname, out_fname):
    pc_with_color= np.load(in_fname)
    xyz = pc_with_color[:,:3]
    if pc_with_color.shape[-1] == 6:
        colors = pc_with_color[:, 3:]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)
    pc.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(out_fname, pc)
    

if __name__ == "__main__":
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    _, _, files = next(os.walk(in_dir))
    file_count = len(files)
    for i in tqdm(range(file_count)):
        name = str(i).zfill(5) 
        in_name = in_dir + "/" + name + "0.npy"
        print("read {}".format(in_name))
        out_name = out_dir + "/" + str(i) + ".pcd"
        npy_to_pcd(in_name, out_name)
        


    
