#from evo.core.trajectory import PoseTrajectory3D
#from evo.tools import file_interface
#from scipy.spatial.transform import Rotation 
import numpy as np
import argparse

def T_change_of_basis(seq_ind):
    T = np.eye(4)
    if seq_ind == "02" or seq_ind == "00":
        # calib_Tr_cam_to_velo.txt
        #T[:3,:3] = np.array([7.967514e-03, -9.999679e-01, -8.462264e-04, -2.771053e-03, 8.241710e-04, -9.999958e-01, 9.999644e-01, 7.969825e-03, -2.764397e-03]).reshape((3,3))
        #T[:3, 3] = np.array([-1.377769e-02 -5.542117e-02 -2.918589e-01])

        # calib.txt
        T[:3,:3] = np.array([4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03,  -7.210626507497e-03, 8.081198471645e-03, -9.999413164504e-01, 9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03 ]).reshape((3,3))
        T[:3, 3] = np.array([-1.198459927713e-02, -5.403984729748e-02, -2.921968648686e-01])

        
    elif seq_ind == "07" or seq_ind == "05" or seq_ind == "06" or seq_ind == "09" or seq_ind == "08":
        # calib_Tr_cam_to_velo.txt        
        # 2011_09_30
        #T[:3,:3] = np.array([7.027555e-03, -9.999753e-01, 2.599616e-05, -2.254837e-03, -4.184312e-05, -9.999975e-01, 9.999728e-01, 7.027479e-03, -2.255075e-03]).reshape((3,3))
        #T[:3, 3] = np.array([-7.137748e-03, -7.482656e-02, -3.336324e-01])

        # calib.txt
        T[:3,:3] = np.array([-1.857739385241e-03, -9.999659513510e-01, -8.039975204516e-03, -6.481465826011e-03, 8.051860151134e-03, -9.999466081774e-01, 9.999773098287e-01, -1.805528627661e-03, -6.496203536139e-03]).reshape((3,3))
        T[:3, 3] = np.array([ -4.784029760483e-03,  -7.337429464231e-02, -3.339968064433e-01])
    else:
        T = None
    return T


def pose_kitti_format_change_of_basis(original_fname, output_fname, seq_ind,
                                      start_ind, end_ind):
    original = np.loadtxt(original_fname, dtype=float)
    print("read file {} with shape {}".format(original_fname, original.shape))

    T_b = T_change_of_basis(seq_ind)
    
    with open(output_fname, 'w') as f:
        if len(original.shape) == 1:
            max_r = 1
        else:
            max_r = original.shape[0]
        counter = 0
        for r in range(max_r):
            if r < start_ind:
                continue
            if end_ind > start_ind and r == end_ind:
                break
            pose = np.eye(4)
            if len(original.shape) > 1:
                rows = original[r, :].size // 4
                p = original[r, :].reshape((-1,4))
                pose[:rows, :] = p                
            else:
                p = original[:].reshape((-1,4))
                pose[:rows, :] = p
            T = T_b @ pose @ np.linalg.inv(T_b)
            #T = np.linalg.inv(T_b) @ pose @ (T_b)
            counter += 1
            for r in range(3):
                for c in range(4):
                    if (r==2 and c == 3):
                        f.write("{}\n".format(T[r, c]))
                    else:
                        f.write("{} ".format(T[r, c]))

        print("Just wrote {} lines".format(counter))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("poses_file", help="input pose path file in xyzq format")
    parser.add_argument(
        "trajectory_out", help="output file path for trajectory in kitti format")
    parser.add_argument("seq_ind", type=str)
    parser.add_argument("start_ind", nargs='?', type=int, default=0, help="starting index")
    parser.add_argument("end_ind", nargs='?', type=int, default=-1, help="last index")

    args = parser.parse_args()
    trajectory = pose_kitti_format_change_of_basis(args.poses_file, args.trajectory_out, args.seq_ind, args.start_ind, args.end_ind)

