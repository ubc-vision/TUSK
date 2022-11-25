import torch
import argparse
import numpy as np
import os
from tqdm import tqdm
import argparse
import h5py
from PIL import Image, ImageDraw
from utils.config import get_config

# we only predict location, use fixed size to compute IOU.
# use same size as in MNIST paper
Detection_Size = 20

def calc_acc(dataset_dir,viz_dir,max_label=10):
	label_gt_dict = h5py.File(os.path.join(dataset_dir,'label_gt.h5'), 'r')
	kp_gt_dict = h5py.File(os.path.join(dataset_dir,'keypoints_gt.h5'), 'r')
	kp_dict = h5py.File(os.path.join(dataset_dir,'keypoints.h5'), 'r')
	label_dict = h5py.File(os.path.join(dataset_dir,'label.h5'), 'r')
	image_dict = h5py.File(os.path.join(dataset_dir,'images.h5'), 'r')
	group_list = list(kp_dict.keys())[:1]
	prototypes_count = [[0 for _ in range(11)] for _ in range(max_label)]
	num_det = 0 
	num_corr_det = 0
	acc = []
	count = 0
	viz_num = 10
	for group in group_list:
		key_list = list(kp_dict[group].keys())
		for key in tqdm(key_list):
			num_det = num_det + 9
			img = np.array(image_dict[group][key])
			img= (np.transpose(img, (1, 2, 0))*255).astype('uint8')
			size,_,_ = img.shape

			kp = kp_dict[group][key][:]
			
			kp_gt = kp_gt_dict[group][key][:]

			kp[:,2] = Detection_Size/size
			label = label_dict[group][key][:]
			label_gt = label_gt_dict[group][key][:]
			kp_min = np.expand_dims(kp[:,:2]-kp[:,[2]]/2,axis=1)
			kp_max = np.expand_dims(kp[:,:2]+kp[:,[2]]/2,axis=1)

			kp_gt_min = np.expand_dims(kp_gt[:,:2]-kp_gt[:,[2]]/2,axis=0)
			kp_gt_max = np.expand_dims(kp_gt[:,:2]+kp_gt[:,[2]]/2,axis=0)

			botleft = np.maximum(kp_min, kp_gt_min)
			topright = np.minimum(kp_max, kp_gt_max)
			inter = np.prod((topright - botleft)*((topright - botleft)>0), axis=2)
			area_K = np.prod(kp_max - kp_min, axis=2)
			area_Kg = np.prod(kp_gt_max - kp_gt_min, axis=2)
			union = area_K + area_Kg - inter
			iou = inter / union
			k,_ = kp_gt.shape

			for i in range(k):
				iou_max_id = np.argmax(iou[:,i])
				iou_max = np.max(iou[:,i])
				if iou_max>0.5:
					num_corr_det = num_corr_det + 1
					prototypes_count[label[iou_max_id]][label_gt[i]] = prototypes_count[label[iou_max_id]][label_gt[i]] +1
					iou[iou_max_id,:] = 0

	num_acc_both = np.sum(np.max(prototypes_count,axis=1))
	prototype_mapping = np.argmax(prototypes_count,axis=1)
	acc = num_acc_both/num_det
	cls_acc = num_acc_both/num_corr_det
	det_acc = num_corr_det/num_det
	print('acc: {}'.format(acc))	
	print('cls_acc: {}'.format(cls_acc))
	print('det_acc: {}'.format(det_acc)	)

	for group in group_list:
		key_list = list(kp_dict[group].keys())
		for key in tqdm(key_list):
			kp = kp_dict[group][key][:]
			kp_gt = kp_gt_dict[group][key][:]
			label = label_dict[group][key][:]
			label_gt = label_gt_dict[group][key][:]
			img = np.array(image_dict[group][key])
			img= (np.transpose(img, (1, 2, 0))*255).astype('uint8')
			size,_,_ = img.shape
			kp = kp *size
			kp_gt = kp_gt*size
			kp[:,2] = Detection_Size
			if count<viz_num:
				# draw gt image
				img_draw = Image.fromarray(img, 'RGB')
				draw = ImageDraw.Draw(img_draw)
				for _label,_kp in zip(label_gt,kp_gt):
					draw.rectangle([_kp[0]-_kp[2]/2,_kp[1]-_kp[2]/2,_kp[0]+_kp[2]/2,_kp[1]+_kp[2]/2], outline=(0, 255, 0))
					draw.text(tuple(_kp[:2]-_kp[2]/2), '{}'.format(_label-1), fill=(255,255,0))
				img_draw.save(os.path.join(viz_dir,key+"_gt.png"))

				# draw infer image
				img_draw = Image.fromarray(img, 'RGB')
				draw = ImageDraw.Draw(img_draw)
				for _label,_kp in zip(label,kp):
					draw.rectangle([_kp[0]-_kp[2]/2,_kp[1]-_kp[2]/2,_kp[0]+_kp[2]/2,_kp[1]+_kp[2]/2], outline=(255,0, 0))
					draw.text(tuple(_kp[:2]-_kp[2]/2.5), '{}'.format(prototype_mapping[_label]-1), fill=(255,255,0))
				img_draw.save(os.path.join(viz_dir,key+"_infer.png"))
			count = count+1



def main():
	config = get_config()
	mode = config.mode

	torch.set_printoptions(profile="full")
	torch.set_printoptions(threshold=5000)
	torch.set_printoptions(precision=10)
	
	# results path
	root_dir = config.metadata.result_root_folder
	dataset = config.data.dataset.name
	name = config.metadata.name

	dataset_dir = os.path.join(root_dir,dataset,name,mode,'output')

	# viz path
	viz_dir = os.path.join(root_dir,dataset,name,mode,'viz')
	if not os.path.isdir(viz_dir):
		os.makedirs(viz_dir)

	calc_acc(dataset_dir,viz_dir)



if __name__ == '__main__':
	main()
