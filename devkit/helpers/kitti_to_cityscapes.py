import numpy as np
import scipy.misc as sp
import matplotlib.pyplot as plt
import os 
join = os.path.join

def kitti_to_cityscapes_instaces(instance_img):
    kitti_semantic = instance_img // 256
    kitti_instance = instance_img %  256
    print(kitti_semantic.max())
    print(kitti_instance.max())

    instance_mask = (kitti_instance > 0)
    cs_instance = (kitti_semantic*1000 + kitti_instance)*instance_mask + kitti_semantic*(1-instance_mask) 
    return cs_instance


if __name__ == '__main__':
    input_dir  = '../../../kitti2015/segmentation/training/'
    output_dir = '../../../kitti2015_cs/gtFine/kitti/'

    semantic_dir = join(input_dir,'semantic/')
    instance_dir = join(input_dir,'instance/')
    out_semantic_dir   = join(output_dir)
    out_instance_dir   = join(output_dir)

    for d in [out_semantic_dir,out_instance_dir] : 
        if not os.path.exists(d):
            os.mkdir(d)

    semantic_file_list = [f for f in os.listdir(semantic_dir) if os.path.isfile(join(semantic_dir,f))]

    for f in semantic_file_list[:4]:
        semantic_img = sp.imread(join(semantic_dir,f))
        instance_img = sp.imread(join(instance_dir,f))


        instance_img = kitti_to_cityscapes_instaces(instance_img)

        out_semantic_filename = join(out_semantic_dir,'kitti_%s_gtFine_labelIds.png'%f[:-4])
        out_instance_filename = join(out_instance_dir,'kitti_%s_gtFine_instanceIds.png'%f[:-4])
        # sp.toimage(semantic_img,mode='L').save(out_semantic_filename)
        # sp.toimage(instance_img,high=np.max(instance_img), low=np.min(instance_img), mode='I').save(out_instance_filename)
