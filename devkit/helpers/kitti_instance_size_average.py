import numpy as np
import scipy.misc as sp
import matplotlib.pyplot as plt
import os 
join = os.path.join
# cs imports 
from helpers.labels  import id2label

def kitti_to_cityscapes_instaces(instance_img):
    kitti_semantic = instance_img // 256
    kitti_instance = instance_img %  256
    print(kitti_semantic.max())
    print(kitti_instance.max())

    instance_mask = (kitti_instance > 0)
    cs_instance = (kitti_semantic*1000 + kitti_instance)*instance_mask + kitti_semantic*(1-instance_mask) 
    return cs_instance


if __name__ == '__main__':
    instanceSizes = {
        "bicycle"    : [] ,
        "caravan"    : [] ,
        "motorcycle" : [] ,
        "rider"      : [] ,
        "bus"        : [] ,
        "train"      : [] ,
        "car"        : [] ,
        "person"     : [] ,
        "truck"      : [] ,
        "trailer"    : [] ,
    }


    for split in ['testing'] :

        instance_dir = join('../../',split,'instance/')
        instance_file_list = [f for f in os.listdir(instance_dir) if os.path.isfile(join(instance_dir,f))]

        for f in instance_file_list[:]:
            instance_img = sp.imread(join(instance_dir,f))
            instclassid_list = np.unique(instance_img)
            for instclassid in instclassid_list:
                instid = instclassid % 256 
                if instid > 0 :
                    classid= instclassid // 256
                    mask = instance_img == instclassid 
                    instance_size = np.count_nonzero(mask)*1.0
                    instanceSizes[id2label[classid].name].append(instance_size)

    print("Average instance sizes : ")
    for className in instanceSizes.keys():
        meanInstanceSize = np.nanmean(instanceSizes[className],dtype=np.float32)
        print('\"%s\"\t: %f,'%(className,meanInstanceSize))


