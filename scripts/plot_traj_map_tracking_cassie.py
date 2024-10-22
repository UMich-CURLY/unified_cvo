#import torch
#import pypose as pp
import numpy as np
import open3d as o3d
import ipdb
import pickle
import sys, os
#from model.utils import create_o3d_pc_from_np
import time

from PIL import Image, ImageFont, ImageDraw
import glob
import cv2

def get_concat_h_blank(im1, im2, color=(0, 0, 0)):
    dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v_blank(im1, im2, color=(0, 0, 0), title=""):
    dst = Image.new('RGB', (max(im1.width, im2.width),
                            im1.height + im2.height),
                            
                    color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    #dst.save('sample-out.jpg')
    return dst


def save_to_gif(iters,
                file_ext="png",
                pc_prefix='',
                img_prefix='',
                file_prefix='',
                skip=1):
    frames = []
    imgs = glob.glob("*."+file_ext)
    font_path = os.path.join('/usr/share/texmf-dist/fonts/truetype/typoland/lato/Lato-Regular.ttf')
    font = ImageFont.truetype(font_path, size=64)    
    for i in range(iters):
        if i % skip != 0:
            continue
        print("process {}".format(i))
        new_pc = Image.open(pc_prefix+str(i)+"."+file_ext)
        color_prefix = str(i).zfill(6)
        color_path = img_prefix+color_prefix+"."+file_ext
        if os.path.exists(color_path):
            print("image {} exist".format(color_path))
            new_color = Image.open(img_prefix+color_prefix+"."+file_ext)
            w = new_color.width
            h = new_color.height
            new_frame = get_concat_v_blank(new_color, new_pc)
            ImageDraw.Draw(new_frame).text(
                (0, new_frame.height - 150),  # Coordinates
                "Stage 1: Frame-to-Frame Semantic Tracking",
                (255, 0, 0),  # Color
                font=font)
            ImageDraw.Draw(new_frame).text(
                (0, new_frame.height - 80),  # Coordinates
                "Stage 2: PGO + Semantic Bundle Adjustment",
                (128, 128, 128),  # Color
                font=font)
        else:
            new_color = Image.new('RGB', ( w, h))
            new_frame = get_concat_v_blank(new_color, new_pc)
            ImageDraw.Draw(new_frame).text(
                (0, new_frame.height - 150),  # Coordinates
                "Stage 1: Frame-to-Frame Semantic Tracking",
                (128, 128, 128),  # Color
                font=font)
            ImageDraw.Draw(new_frame).text(
                (0, new_frame.height - 80),  # Coordinates
                "Stage 2: PGO + Semantic BA",
                (255, 0, 0),  # Color
                font=font)
            

        base_width= 480
        wpercent = (base_width / float(new_frame.size[0]))
        hsize = int((float(new_frame.size[1]) * float(wpercent)))
        new_frame = new_frame.resize((base_width, hsize), Image.Resampling.LANCZOS)

        #new_frame.save('sample-out.jpg')

        frames.append(new_frame)

    # Save into a GIF file that loops forever
    frames[0].save(file_prefix+'.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=1)

def parse_lc_file(lc_fname):
    return np.genfromtxt(lc_fname, dtype=int, skip_header=1)


def construct_line(pt1, pt2, color):

    print("Let\'s draw a cubic using o3d.geometry.LineSet")
    points = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1],
              [0, 1, 1], [1, 1, 1]]
    lines = [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

def read_kitti_pose(pose_file):
    poses_34 = np.genfromtxt(pose_file)
    poses = np.zeros((poses_34.shape[0], 4, 4))
    for i in range(poses_34.shape[0]):
        poses[i, :, :] = np.eye(4)
        poses[i, :3,:] = poses_34[i, :].reshape((3,4))

    return poses

def visualize_traj(pcd_dir,
                   pose_file,
                   save_image=False,
                   is_show_image=False,
                   step_size=10,
                   start_frame = 0,
                   end_frame=100000):
    
    track_poses = read_kitti_pose(pose_file)


    
    num_frames = track_poses.shape[0]

    
    vis = o3d.visualization.VisualizerWithKeyCallback()

    vis.create_window()
    

    #
    pcs_o3d = []
    pointset = []    
    for i in range(start_frame, min(num_frames, end_frame+1), 1):
        fname = pcd_dir + "/" + str(i*step_size)+".pcd"
        print("read ",fname)
        pc = o3d.io.read_point_cloud(fname) #pc_dict[i]['xyz']
        pcs_o3d.append(pc)

        #color = 
        #pc = create_o3d_pc_from_np(xyz, color)
        track_pose = track_poses[i,:,:]
        num_pts = np.asarray(pc.points).shape[0]
        if num_pts == 0:
            break 
        print("new pc #{}, pose\n{}\n".format(np.asarray(pc.points).shape[0], track_pose))

        p1 = track_poses[i, :3, 3]
        pointset.append(list(p1))
        

        pc.transform(track_pose)
        vis.add_geometry(pc,True)

        
        vis.poll_events()
        vis.update_renderer()
        if save_image:
            if os.path.exists(pcd_dir + "/png_tracking/") == False:
                os.mkdir(pcd_dir + "/png_tracking/")
            vis.capture_screen_image(pcd_dir + "/png_tracking/" + str(i) + ".png" )

        #vis.run()
        #img=cv2.imread(img_dir+lst[i])
        #cv2.imshow("current img", img)
        #cv2.imwrite("img_{}.png".format(i), img)
        #cv2.waitKey(0)
        
        #input("press key to continue, curr frame: "+ str(i*step_size))
        #time.sleep(0.01)

    ################################
    # visualize loop closure edges
    ################################
    #input("Press Enter to add loop closures")
    '''
    lines = []
    for i in range(lc_pairs.shape[0]):
        id1 = lc_pairs[i][0]
        id2 = lc_pairs[i][1]
        last_inds = len(pointset)-2, len(pointset)-1
        lines.append([last_inds[0], last_inds[1]])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pointset)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([[1,0,0] for _ in range(len(lines))])
    o3d.visualization.draw_geometries([line_set])
    mat = o3d.visualization.rendering.Material()
    mat.shader = "unlitLine"
    mat.line_width = 5  # note that this is scaled with respect to pixels,
                        # so will give different results depending on the
                        # scaling values of your system
    #parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2023-11-27-05-31-08.json")
    #ctr.convert_from_pinhole_camera_parameters(parameters)
    vis.poll_events()
    vis.update_renderer()
    if save_image:
        vis.capture_screen_image(str(i) + ".png" )

    time.sleep(2)
    ################################
    # visualize the ba results
    ################################
    for i in range( num_frames):
        ba_pose = ba_poses[i,:,:]
        track_pose = track_poses[i, :, :]

        pose_update = ba_pose @ np.linalg.inv(track_pose)
        pcs_o3d[i].transform(pose_update)
        vis.update_geometry(pcs_o3d[i])#,False)
        
        vis.poll_events()
        vis.update_renderer()
    if save_image:
        if os.path.exists(pcd_dir + "/png_ba/") == False:
            os.mkdir(pcd_dir + "/png_ba/")
        vis.capture_screen_image(pcd_dir + "/png_ba/" + str(i) + ".png" )            
            #vis.capture_screen_image(str(i) + ".png" )
    '''
    vis.run()
    #save_to_gif(num_frames,
    #            img_prefix='img_',
    #           file_prefix='sfm_lab_room_1')

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)


if __name__ == '__main__':


    #####################
    # stack pc and take screenshots
    #####################
    #pcd_folder = "kitti_lidar_pcd_cam_frame_05/" #sys.argv[1] ### cam frame
    pcd_folder = "../cassie_wavefield_short/intensity_pcd/" #"../cassie_wavefield/segmented_pcd"
    track_pose_file = "../cassie_wavefield_short_oct7/ba.txt" #"../cassie_wavefield/cassie_pose.txt" #"lidar_05_cvo_odom_old.txt" #"results/cvo_intensity_lidar_jun09/05.txt" #sys.argv[2]

    start_frame = 0
    end_frame = 100000
    #if not os.path.exists(pcd_folder+"/png/"):
    #    os.mkdir(pcd_folder+"/png")
    visualize_traj(pcd_folder,
                   track_pose_file,
                   True, #True # is saving image
                   False,
                   step_size=10,
                   start_frame=start_frame,
                   end_frame=end_frame
                   )


    ########################
    # make gif
    ########################
    '''
    save_to_gif(4000,
                file_ext="png",
                pc_prefix="kitti_lidar_pcd_lidar_frame_downsample_05/png/",
                img_prefix="/run/media/rayzhang/Samsung_T51/kitti_stereo/dataset/sequences/05/image_2/",
                file_prefix="kitti05_video_fast",
                skip=20)
    '''

