import open3d as o3d
import sys, os
import numpy as np

pcds = []
for i in range(len(sys.argv)):
    if i == 0: 
        continue
    pcd = o3d.io.read_point_cloud(sys.argv[i])
    print(pcd)
    print(np.asarray(pcd.points))
    pcds.append(pcd)
o3d.visualization.draw_geometries(pcds)
