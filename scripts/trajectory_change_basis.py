#from evo.core.trajectory import PoseTrajectory3D
#from evo.tools import file_interface
#from scipy.spatial.transform import Rotation 
import numpy as np
import argparse

def T_change_of_basis(seq_ind):
    T = np.eye(4)
    if seq_ind == "07":
        T[:3,:3] = np.array([7.027555e-03, -9.999753e-01, 2.599616e-05, -2.254837e-03, -4.184312e-05, -9.999975e-01, 9.999728e-01, 7.027479e-03, -2.255075e-03]).reshape((3,3))
        T[:3, 3] = np.array([-7.137748e-03, -7.482656e-02, -3.336324e-01])
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
                p = original[r, :].reshape((4,4))
            else:
                p = original[:].reshape((4,4))
            pose[:, :] = p

            T = T_b @ pose @ np.linalg.inv(T_b)
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

