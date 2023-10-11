import numpy as np
import open3d as o3d
import sys, os

def visual(fname1, fname2):
    pc1 = np.load(fname1)
    print(pc1.shape)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc1)

    pc2 = np.load(fname2)
    print(pc2.shape)    
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pc2)

    o3d.visualization.draw_geometries([pcd1, pcd2])

def visual(fname1):
    pc1 = np.load(fname1)
    print(pc1.shape)
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc1)

    o3d.visualization.draw_geometries([pcd1])


if __name__ == "__main__":
    if len(sys.argv) > 2:
        visual(sys.argv[1], sys.argv[2])
    else:
        visual(sys.argv[1])
    
