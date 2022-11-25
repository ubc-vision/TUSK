import torch 
import os
import numpy as np
import skimage
from tqdm import tqdm
import h5py
from datasets.dataset_utils import gen_image_pairs
import torch.nn.functional as F

color_dict = {
  '1-0-0': 0,
  '0-1-0': 1,
  '0-0-1': 2,
  '0-1-1': 3,
  '1-0-1': 4,
  '1-1-0': 5

}

class MultiObject(torch.utils.data.Dataset):
    def __init__(self, config, mode, eqvar, evaluate=False):
        self.eqvar = eqvar
        self.dataset_name = config.dataset.sub_name
        self.evaluate = evaluate
        # preprocess
        if self.dataset_name == 'CLEVR':
            # only keep object less than 6
            self.background_entities = 1
            self.root_dir = os.path.join(config.dataset.dir,'multi-object-datasets/clevr_with_masks')
            self.h5 = 'clevr_with_masks_train.h5'
            self.h5_path  = os.path.join(self.root_dir,self.h5)
            with h5py.File(self.h5_path, "r") as self.dataset:
                self.idx_list = [key for idx, (key, value) in tqdm(enumerate(self.dataset['color'].items())) if value[7] ==0 and idx<70000]
                if mode == 'train':
                    self.idx_list = self.idx_list[:-10000]
                else:
                    self.idx_list = self.idx_list[-10000:]
                self.len = len(self.idx_list)                  
        elif self.dataset_name == 'tetrominoes':
            self.background_entities = 1
            self.root_dir = os.path.join(config.dataset.dir,'multi-object-datasets/tetrominoes')
            self.h5 = 'tetrominoes_train.h5'
            self.h5_path  = os.path.join(self.root_dir,self.h5)
            with h5py.File(self.h5_path, "r") as self.dataset:
                self.idx_list = [key for idx, (key, _) in tqdm(enumerate(self.dataset['color'].items())) if idx<60000]
                # self.idx_list = [key for idx, (key, value) in tqdm(enumerate(self.dataset['color'].items()))]
                if mode == 'train':
                    self.idx_list = self.idx_list[:-10000]
                elif mode == 'test':
                    self.idx_list = self.idx_list[-10000:]
                self.len = len(self.idx_list)   
        self.dataset = None
        self.num_kp = 0
        self.image_size =config.preprocess.image_size

    def __len__(self):
        return self.len

    def center_crop(self,image,crop_size):
        shape = image.shape
        s1 = (shape[1]-crop_size)//2
        s2 = (shape[2]-crop_size)//2
        return image[:, s1:s1+crop_size, s2:s2+crop_size]

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.h5_path, "r")

        img = np.asarray(self.dataset['image'][self.idx_list[idx]])
        img = np.moveaxis(img, 2, 0)
        shape = img.shape # C,H,W 
        
        # center crop on CLEVR
        if self.dataset_name == 'CLEVR':
            img = self.center_crop(img,192)
        img = torch.FloatTensor(img) / 255.
        img = F.interpolate(torch.unsqueeze(img,0), size=(self.image_size,self.image_size)).squeeze()

        labels = torch.zeros([self.num_kp],dtype=torch.long)
        keypoints = torch.zeros([self.num_kp,2])
        if self.eqvar:
            img, masks, tps_grid = gen_image_pairs(img.unsqueeze(0), False, scal=0.95, scal_var=0.02, tps_scal=0.02, off_scal=0.05, rot_scal=0.1)
        else:
            masks = torch.zeros([1])
            tps_grid = torch.zeros([1])        
        if not self.evaluate:
            return img, masks, tps_grid, keypoints, labels, self.idx_list[idx]
        else:
            # generate labels
            raw_masks = np.asarray(self.dataset['mask'][self.idx_list[idx]])
            gt_masks = np.zeros(( 1, shape[1], shape[2]), dtype='int')
            cond = np.where(raw_masks[:, :, :, 0] == 255, True, False)
            num_entities = cond.shape[0]
            for o_idx in range(self.background_entities, num_entities):
                gt_masks[cond[o_idx:o_idx+1, :, :]] = o_idx + 1
            gt_masks = torch.FloatTensor(gt_masks)
            if self.dataset_name == 'CLEVR':
                gt_masks = self.center_crop(gt_masks,192)
                gt_masks = torch.FloatTensor(gt_masks)
                gt_masks = F.interpolate(torch.unsqueeze(gt_masks,0), size=(112,112)).squeeze()
                gt_masks = gt_masks.type(torch.LongTensor)
                shape_labels = torch.LongTensor(np.asarray(self.dataset['shape'][self.idx_list[idx]])[1:])-1
                color_labels = torch.LongTensor(np.asarray(self.dataset['color'][self.idx_list[idx]])[1:])-1
                size_labels = torch.LongTensor(np.asarray(self.dataset['size'][self.idx_list[idx]])[1:])-1
                labels = shape_labels *16 + color_labels *2 +size_labels
            elif self.dataset_name == 'tetrominoes':
                gt_masks = torch.FloatTensor(gt_masks)
                gt_masks = F.interpolate(torch.unsqueeze(gt_masks,0), size=(112,112)).squeeze()
                gt_masks = gt_masks.type(torch.LongTensor)
                shape_labels = torch.LongTensor(np.asarray(self.dataset['shape'][self.idx_list[idx]])[1:])-1
                color_labels = torch.LongTensor([color_dict['-'.join([str(int(c)) for c in list(cc)])] for cc in list(self.dataset['color'][self.idx_list[idx]])[1:]])
                labels = shape_labels * 6 + color_labels
                size_labels = torch.LongTensor(np.zeros(3))

            return img, gt_masks, masks, tps_grid, keypoints, shape_labels, color_labels, size_labels, labels, self.idx_list[idx]
            
