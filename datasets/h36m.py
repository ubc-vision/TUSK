import torch 
import os
import numpy as np
import skimage
from tqdm import tqdm
import h5py
from datasets.dataset_utils import gen_image_pairs


class H36M(torch.utils.data.Dataset):
    def __init__(self, config, mode, eqvar):
        self.eqvar = eqvar
        self.root_dir = os.path.join(config.dataset.dir,'crop_foreground_h36m')

        self.image_paths = np.genfromtxt(self.root_dir + '/{}_img_list'.format(mode) + '.txt', delimiter=',', dtype='str', encoding='utf-8')
        print(self.root_dir + '/{}_img_list'.format(mode) + '.txt')
        self.labels = np.tile(np.arange(32),(self.image_paths.size,1))
        self.anno_file_path = os.path.join(self.root_dir,'cropped_pose.h5')
        self.h5f = None
        
        self.image_size =config.preprocess.image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx): 
        if self.h5f == None:
            self.h5f = h5py.File(self.anno_file_path,'r')

        keypoints = torch.from_numpy((np.array(list(self.h5f[self.image_paths[idx]])).reshape(32,2)))

        img_name = os.path.join(self.root_dir, self.image_paths[idx])
        image = skimage.io.imread(img_name)

        image_shape_min = min(image.shape[0],image.shape[1])
        stride = self.image_size/image_shape_min
        image = skimage.transform.resize(image,(int(image.shape[0]*stride),int(image.shape[1]*stride)))

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        image = torch.clamp(image, 0.0, 1.0)


        keypoints[:,0] = keypoints[:,0]*stride
        keypoints[:,1] = keypoints[:,1]*stride

        labels = torch.from_numpy(self.labels[idx])


        if self.eqvar:
            images, masks, tps_grid =  gen_image_pairs(image.unsqueeze(0),False, scal=0.95, scal_var=0.1, tps_scal=0.05, off_scal=0.1, rot_scal=0.25, padding='Constant')
        else:
            images = image
            masks = torch.zeros([1])
            tps_grid = torch.zeros([1])
        return images, masks, tps_grid, keypoints, labels, self.image_paths[idx].replace('/','-').replace(' ','-')