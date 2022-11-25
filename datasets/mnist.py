import torch 
import os
import numpy as np
import skimage
from tqdm import tqdm
from datasets.dataset_utils import gen_image_pairs

class MNIST(torch.utils.data.Dataset):
    def __init__(self, config,mode,eqvar):
        self.eqvar = eqvar
        root_dir = config.dataset.dir+'/'+config.dataset.name+'/'
        self.img_paths =[ os.path.join(root_dir,img_path) for img_path in np.genfromtxt(root_dir + mode + '.txt', delimiter=',', dtype='str', encoding='utf-8')]
        self.labels = np.genfromtxt(root_dir + mode +'_labels.txt', delimiter=',', dtype='int', encoding='utf-8')
        self.keypoints = np.load(root_dir + mode +'_keypoints.npy')
        self.image_size =config.preprocess.image_size


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        image = skimage.io.imread(self.img_paths[idx])


        stride = self.image_size/image.shape[0]

        image = skimage.transform.resize(image,(self.image_size,self.image_size))

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        image = torch.clamp(image, 0.0, 1.0)

        keypoints = torch.from_numpy(self.keypoints[idx].copy())

        keypoints[:,2] = keypoints[:,2] * 2.0
        keypoints = torch.cat((keypoints,keypoints[:,[2]]), axis=-1)
        keypoints = keypoints*stride

        labels = torch.from_numpy(self.labels[idx])
        labels=labels+1
        if self.eqvar:
            image, masks, tps_grid = gen_image_pairs(image.unsqueeze(0),False, scal=0.95, scal_var=0.05, tps_scal=0.1, off_scal=0.01, rot_scal=0.25,padding='Constant',const = 0)
        else:
            masks = torch.zeros([1])
            tps_grid = torch.zeros([1])   

        return image, masks, tps_grid, keypoints, labels, self.img_paths[idx].split('/')[-1]
