import torch 
import os
import numpy as np
import skimage
from tqdm import tqdm
import h5py
from datasets.dataset_utils import gen_image_pairs

class CelebA(torch.utils.data.Dataset):
    def __init__(self, config, mode, eqvar):
        self.eqvar = eqvar

        root_dir = os.path.join(config.dataset.dir,'CelebA')
        
        # valid image paths (remove image with small face)
        iou = 0.3
        img_names = np.genfromtxt(os.path.join(root_dir, 'Eval', config.dataset.name + '_' + mode + '.txt'), delimiter=',', dtype='str', encoding='utf-8')
        img_sizes = [l.split() for l in open(os.path.join(root_dir, 'Anno/list_imsize_celeba.txt')) if len(l.split())==3 and l[:8] != 'image_id']
        img_size_dict = {x[0]:[int(x[1]), int(x[2])] for x in img_sizes}
        bboxes = [l.split() for l in open(os.path.join(root_dir, 'Anno/list_bbox_celeba.txt')) if len(l.split())==5 and l[:8] != 'image_id']
        bbox_dict = {x[0]:[int(x[1]), int(x[2]), int(x[3]), int(x[4])] for x in bboxes}
        self.image_paths = [os.path.join(root_dir,'img_celeba', img_name) for img_name in img_names if (bbox_dict[img_name][2]*bbox_dict[img_name][3])/ (img_size_dict[img_name][0]*img_size_dict[img_name][1]) >iou]

        # labels
        self.labels = np.tile(np.array([1,2,3,4,5]),(len(self.image_paths),1))

        # annotations
        anno = {}
        anno_file = 'Anno/list_landmarks_celeba.txt'
        with open(os.path.join(root_dir,anno_file)) as fp:
            Lines = fp.readlines()
            for line in Lines:
                if line[7:10] == 'jpg':
                    line_list = line.split(' ')
                    key = line_list[0]
                    num_list = [int(num) for num in line_list[1:] if (num != ' ' and num !='') ]
                    anno[key]= np.array([[num_list[0],num_list[1]],[num_list[2],num_list[3]],[num_list[4],num_list[5]],[num_list[6],num_list[7]],[num_list[8],num_list[9]]])      
        self.keypoints = np.zeros([len(self.image_paths),5,2])
        for idx, image_path in enumerate(self.image_paths):
            key =  image_path.split('/')[-1]
            self.keypoints[idx,:,:] = anno[key]

        self.image_size =config.preprocess.image_size


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        image = skimage.io.imread(self.image_paths[idx])

        stride_x = self.image_size/image.shape[1]
        stride_y = self.image_size/image.shape[0]
        image = skimage.transform.resize(image,(self.image_size,self.image_size))

        image = torch.from_numpy(image).permute(2, 0, 1).float()
        image = torch.clamp(image, 0.0, 1.0)

        keypoints = torch.from_numpy(self.keypoints[idx].copy())

        keypoints[:,0] = keypoints[:,0]*stride_x
        keypoints[:,1] = keypoints[:,1]*stride_y

        labels = torch.from_numpy(self.labels[idx])

        if self.eqvar:
            image, masks, tps_grid = gen_image_pairs(image.unsqueeze(0),False)
        else:
            masks = torch.zeros([1])
            tps_grid = torch.zeros([1])   

        return image, masks, tps_grid, keypoints, labels, os.path.splitext(os.path.basename(self.image_paths[idx]))[0]