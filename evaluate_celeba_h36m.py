import torch
import numpy as np
import os
from tqdm import tqdm
import h5py
from PIL import Image, ImageDraw
from matplotlib import cm
import matplotlib.pyplot as plt
import argparse
from utils.torch_utils import to_gpu
from utils.config import print_config, get_config
JOINT_SYMMETRY = [[6,1],[7,2],[8,3],[9,4],[10,5],[16 ,24],[17,25],[18,26],[19,27],[20,28],[21,29],[22,30],[23,31]]

class landmark_dataset(torch.utils.data.Dataset):
	def __init__(self, root_dir, dataset,name,mode,num_kps,num_prototypes):
		self.root_dir = root_dir+'/'+dataset+'/'+name+'/'+mode+'/output/'
		self.image_dict = h5py.File(self.root_dir+'images.h5', 'r')
		self.kp_dict = h5py.File(self.root_dir+'keypoints.h5', 'r')
		self.kp_gt_dict = h5py.File(self.root_dir+'keypoints_gt.h5', 'r')
		self.label_dict = h5py.File(self.root_dir+'label.h5', 'r')
		self.feature_dict = h5py.File(self.root_dir+'features.h5', 'r')
		self.keys = {}
		for key,item in self.image_dict.items():
			self.keys[key] = list(item.keys())
		self.size = sum([len(item.keys()) for _, item in self.image_dict.items()])
		self.num_kps = num_kps
		self.num_prototypes = num_prototypes
		if dataset == 'H36M':
			self.symmetry = True
			self.joint_symmetry = np.array(JOINT_SYMMETRY)
		else:
			self.symmetry = False
		
		self.image_dict = None
		self.kp_dict = None
		self.kp_gt_dict = None
		self.label_dict = None	
		self.feature_dict = None	

	def __len__(self):
		return self.size

	def __getitem__(self, idx):
		if self.image_dict is None:
			self.image_dict = h5py.File(self.root_dir+'images.h5', 'r')
			self.kp_dict = h5py.File(self.root_dir+'keypoints.h5', 'r')
			self.kp_gt_dict = h5py.File(self.root_dir+'keypoints_gt.h5', 'r')
			self.label_dict = h5py.File(self.root_dir+'label.h5', 'r')		
			self.feature_dict = h5py.File(self.root_dir+'features.h5', 'r')
		
		key1 = "{:05d}".format(int(idx/10000))
		key = self.keys[key1][int(idx%10000)]
		kps = self.kp_dict[key1][key][:]
		kps_gt = self.kp_gt_dict[key1][key][:]
		labels = self.label_dict[key1][key][:]
		img  = self.image_dict[key1][key][:]

		C, H, W = img.shape 

		if self.symmetry:
			if np.sum(kps_gt[self.joint_symmetry[:,1],0]>kps_gt[self.joint_symmetry[:,0],0])>len(self.joint_symmetry[:,1])*0.67:
				# front facing
				kps_gt_new = kps_gt
			else:
				# rear facing, flip kp
				kps_gt_new = kps_gt.copy()
				kps_gt_new[self.joint_symmetry[:,0],:] = kps_gt[self.joint_symmetry[:,1],:]
				kps_gt_new[self.joint_symmetry[:,1],:] = kps_gt[self.joint_symmetry[:,0],:]
		else:
			kps_gt_new = kps_gt

		# construct sparse matrix
		kps_scatter = torch.zeros([self.num_kps*self.num_prototypes,3])
		counter = [0 for _ in range(self.num_prototypes)]
		for kp, label in zip(kps,labels):
			if counter[label]==0:
				kps_scatter[self.num_kps*label+counter[label],0] = kp[0].item()
				kps_scatter[self.num_kps*label+counter[label],1] = kp[1].item()
				kps_scatter[self.num_kps*label+counter[label],2] = kp[2].item()
			else:
				move_idx=counter[label]
				for i in range(counter[label]-1):
					if W*kps_scatter[self.num_kps*label+i,1]+kps_scatter[self.num_kps*label+i,0] > W*kp[1].item()+kp[0].item():
						move_idx = i
						break
				for i in reversed(range(move_idx,counter[label])):
					kps_scatter[self.num_kps*label+i+1,0] = kps_scatter[self.num_kps*label+i,0]
					kps_scatter[self.num_kps*label+i+1,1] = kps_scatter[self.num_kps*label+i,1]
					kps_scatter[self.num_kps*label+i+1,2] = kps_scatter[self.num_kps*label+i,2]
				kps_scatter[self.num_kps*label+move_idx,0] = kp[0].item()
				kps_scatter[self.num_kps*label+move_idx,1] = kp[1].item()	
				kps_scatter[self.num_kps*label+move_idx,2] = kp[2].item()							
			counter[label] = counter[label] + 1
		kps_gt_new = torch.from_numpy(kps_gt_new)
		img = torch.from_numpy(img)

		return kps_scatter, kps_gt_new, torch.zeros(0), img, key, torch.from_numpy(kps),torch.from_numpy(labels)

class LinearProjection(torch.nn.Module):
	def __init__(self,num_kps, num_prototypes,output_dim):
		super(LinearProjection, self).__init__()
		self.mlp = torch.nn.Linear(num_kps*num_prototypes*2, output_dim, bias=False)
	def forward(self,x):
		y=self.mlp(x)
		return y

def main():

	config = get_config()
	
	# use MAFL subset for evaluation
	if config.data.dataset.name.startswith('Celeb'):
		config.data.dataset.name = 'MAFL'


	if config.data.dataset.name.startswith('MAFL'):
		num_output = 5
		num_epoch = 5
	elif config.data.dataset.name.startswith('H36M'):  
		num_output = 32
		num_epoch = 2
	network_type = 'LinearProjection'

	# init dataloader
	B = 64
	loader = torch.utils.data.DataLoader(landmark_dataset(config.metadata.result_root_folder,config.data.dataset.name, config.metadata.name, config.mode, config.model.top_k, config.model.num_cluster), B, shuffle=True, num_workers=8, pin_memory=False)
	# init network
	network = LinearProjection(config.model.top_k, config.model.num_cluster,num_output*2).cuda( )

	
	# color map for viz
	color_map = cm.get_cmap('tab20', 33)
	color_map = color_map(np.linspace(0, 1, 33))
	color_map = (color_map[:,:3]*255).astype(int)
	
	model_dir = os.path.join(config.metadata.result_root_folder,config.data.dataset.name,config.metadata.name)
	if not os.path.isdir(model_dir):
		os.makedirs(model_dir)

	viz_dir = os.path.join(config.metadata.result_root_folder,config.data.dataset.name, config.metadata.name,config.mode,'viz')
	if not os.path.isdir(viz_dir):
		os.makedirs(viz_dir)
	if config.mode == 'train':
		# init solver
		network_solver = torch.optim.Adam(network.parameters(), lr=1e-2)
		
		#train loop
		counter = 0
		for _ in tqdm(range(num_epoch)):
			for x in tqdm(loader, smoothing=0.1):
				kps_scatter, kps_gt,_,imgs,keys, kps, labels = to_gpu(x,0)
				B,_,_ = kps_scatter.shape

				kps_infer = network(kps_scatter[:,:,:2].reshape(B,-1))
				# loss = torch.mean(torch.square(kps_infer.view(B,-1,2)-kps_gt[:,:,:2])*kps_gt[:,:,[2]])
				loss = torch.mean(torch.square(kps_infer.view(B,-1,2)-kps_gt[:,:,:2]))
				print(loss)
				network_solver.zero_grad()
				loss.backward()
				network_solver.step()
				if counter%200 ==0:
					torch.save({'model': network.state_dict()}, os.path.join(model_dir, '{}_model'.format(network_type)))
				counter = counter + 1
	
	elif config.mode=='test' or config.mode=='valid':

		mse = 0
		counter = 0

		checkpoint = torch.load(os.path.join(model_dir, '{}_model'.format(network_type)))
		network.load_state_dict(checkpoint['model'])	

		num_viz = 10
		for idx,x in tqdm(enumerate(loader), smoothing=0.1):

			kps_scatter, kps_gt, _, imgs, keys, kps, labels = to_gpu(x,0)
			B,_,_ = kps_scatter.shape
			kps_infer = network(kps_scatter[:,:,:2].reshape(B,-1))	
			
			kps_infer_xy = kps_infer.view(kps_infer.shape[0],-1,2).cpu()


			kps_gt_xy = kps_gt.cpu()[:,:,:2] 
			kps_xy = kps.cpu()[:,:,:2] 
			labels = labels.cpu()

			ocu_dist = torch.sqrt(torch.sum((kps_gt_xy[:,0,:] - kps_gt_xy[:,1,:])*(kps_gt_xy[:,0,:] - kps_gt_xy[:,1,:]),dim=1))

			diff_dist = torch.sqrt(torch.sum((kps_gt_xy - kps_infer_xy)*(kps_gt_xy - kps_infer_xy),dim=2))#*kps_gt_conf)

			if  config.data.dataset.name.startswith('MAFL'):
				diff_dist = diff_dist/ocu_dist.unsqueeze(-1)

			mse = mse + torch.sum(diff_dist)
			counter = counter+ B*num_output
	
			if idx<num_viz:
				kp_infer_xy_0 = kps_infer_xy[0]
				img = imgs.cpu().detach().numpy()[0]
				key = keys[0]
				kp_infer_xy_0 = kp_infer_xy_0 * imgs.shape[-1]
				img= (np.transpose(img, (1, 2, 0))*255).astype('uint8')
				img_draw = Image.fromarray(img, 'RGB')
				draw = ImageDraw.Draw(img_draw)
				for kp in kp_infer_xy_0:
					color =tuple([0,255,0])
					draw.ellipse([tuple(kp-3),tuple(kp+3)], outline=color,width=2)		
				img_draw.save(os.path.join(viz_dir,key+'_infer.png'))

				kps_xy_0 = kps_xy[0]
				kps_xy_0 = kps_xy_0 * imgs.shape[-1]
				labels_0 = labels[0]
				img_draw = Image.fromarray(img, 'RGB')
				draw = ImageDraw.Draw(img_draw)

				for kp,label in zip(kps_xy_0,labels_0):
						draw.ellipse([tuple(kp-3),tuple(kp+3)], outline=tuple(color_map[label]),width=2)
				img_draw.save(os.path.join(viz_dir,key+".png"))

				kps_gt_xy_0 = kps_gt_xy[0]
				kps_gt_xy_0 = kps_gt_xy_0 * imgs.shape[-1]
				img_draw = Image.fromarray(img, 'RGB')
				draw = ImageDraw.Draw(img_draw)
				for kp in kps_gt_xy_0:
					color =tuple([255,0,0])
					draw.ellipse([tuple(kp-3),tuple(kp+3)], outline=color,width=2)		
				img_draw.save(os.path.join(viz_dir,key+"_gt.png"))

		print('mse normalizd : {}'.format(mse/counter*100))

if __name__ == '__main__':
	main()
