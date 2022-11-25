import torch
import numpy as np
import os
from tqdm import tqdm
import h5py
from utils.config import print_config, get_config
from utils.torch_utils import to_gpu
from datasets.dataset import dataset
from models.network import NetWork
from PIL import Image, ImageDraw
from matplotlib import cm


def main():
    num_samples = -1
    config = get_config()

    config.training.loss.feature.eqvar.activate = False
    config.training.batch_size = 20

    if config.data.dataset.name == 'CelebA':
        config.data.dataset.name = 'MAFL'
    print_config(config)

    # for mode in ['test']:
    for mode in ['train','test']:
        if config.model.num_cluster <= 10: 
            color_map = cm.get_cmap('tab10', 10)
            color_map = color_map(np.linspace(0, 1, 10))
        elif config.model.num_cluster <= 20:  
            color_map = cm.get_cmap('tab20', 20)
            color_map = color_map(np.linspace(0, 1, 20))
        else:
            color_map = cm.get_cmap('gist_rainbow')
            color_map = color_map(np.linspace(0, 1, config.model.num_cluster))
        color_map = (color_map[:,:3]*255).astype(int)
        viz_folder_hq = '{}/{}/{}/{}/viz/'.format(config.metadata.result_root_folder,config.data.dataset.name,config.metadata.name,mode)
        dir_path = '{}/{}/{}/{}/output/'.format(config.metadata.result_root_folder,config.data.dataset.name,config.metadata.name,mode)
        if not os.path.isdir(viz_folder_hq):
            os.makedirs(viz_folder_hq)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        # init data loader
        dataset_tr, _ = dataset(config, mode=mode)

        # init network
        network = NetWork(config.model)

        # load model
        print('Loading saved models: {}'.format(config.metadata.model_dir + '/' + config.metadata.name+'_models'))
        checkpoint = torch.load(os.path.join(config.metadata.model_dir, config.metadata.name + '_models'))
        network.load_state_dict_ddp(checkpoint['model'])
        network = network.cuda()

        num_viz_samples = 100
        counter = 0
        with h5py.File(dir_path+'keypoints.h5', 'w') as f_kp, \
            h5py.File(dir_path+'keypoints_gt.h5', 'w') as f_gt, \
            h5py.File(dir_path+'features.h5','w') as f_feature, \
            h5py.File(dir_path+'images.h5', 'w') as f_img, \
            h5py.File(dir_path+'label.h5', 'w') as f_label, \
            h5py.File(dir_path+'label_gt.h5', 'w') as f_label_gt:
            for idx, x in tqdm(enumerate(dataset_tr), smoothing=0.1):
                if num_samples!=-1 and idx >= num_samples:
                    break

                # get data
                images, _, _, kp_gt, labels_gt, img_names  = to_gpu(x,0)
                B,_,_,_ = images.shape

                # sanity check on data format
                if len(images.shape)!=4 or images.shape[1]!=3:
                    raise RunTimeError('Images does not have dimension (B,3,H,W)')
 
                # run network
                _, keypoints, labels, features_norm, _, _ = network.forward(images, opt_prototypes = False)

                for image,kp,gt,label,label_gt,img_name,feature_norm in zip(images,keypoints,kp_gt, labels,labels_gt,img_names,features_norm):
                    group_name = "{:05d}".format(int(counter/10000))
                    if not group_name in f_gt.keys():
                        f_gt.create_group(group_name)
                        f_img.create_group(group_name)
                        f_kp.create_group(group_name)
                        f_label.create_group(group_name)
                        f_label_gt.create_group(group_name)
                        f_feature.create_group(group_name)
                    kp = kp.cpu().detach()
                    label = label.cpu().detach()
                    feature_norm = feature_norm.cpu().detach()
                    gt = gt.cpu().detach()

                    if config.data.dataset.name.startswith('mnist'):
                        f_gt[group_name][img_name] = np.concatenate([gt.numpy()[:,[0]]/images.shape[-1],gt.numpy()[:,[1]]/images.shape[-2],gt.numpy()[:,[2]]/images.shape[-2]],axis=-1)
                    else:
                        f_gt[group_name][img_name] = np.concatenate([gt.numpy()[:,[0]]/images.shape[-1],gt.numpy()[:,[1]]/images.shape[-2]],axis=-1)
                    
                    f_img[group_name][img_name] = image.cpu().detach().numpy()
                    f_kp[group_name][img_name]= np.concatenate([kp.numpy()[:,[0]]/images.shape[-1],kp.numpy()[:,[1]]/images.shape[-2],kp.numpy()[:,[2]]],axis=-1)
                    f_label[group_name][img_name]= label.numpy().squeeze()
                    f_label_gt[group_name][img_name] = label_gt.cpu().detach().numpy()
                    f_feature[group_name][img_name] = feature_norm.numpy()

                    if counter<num_viz_samples:
                        img_np = (np.transpose(f_img[group_name][img_name], (1, 2, 0))*255).astype('uint8')
                        img = Image.fromarray(img_np, 'RGB')
                        draw = ImageDraw.Draw(img)
                        for kp,label in zip(f_kp[group_name][img_name],f_label[group_name][img_name]):
                            kp[0] = kp[0] * images.shape[-1]
                            kp[1] = kp[1] * images.shape[-2]
                            draw.text(tuple(kp[:2]-5), '{}'.format(label), fill=(255,255,0))
                        img.save(viz_folder_hq + img_name+".png")
                        img = Image.fromarray(img_np, 'RGB')
                        draw = ImageDraw.Draw(img)
                        for kp in f_gt[group_name][img_name]:
                            kp[0] = kp[0] * images.shape[-1]
                            kp[1] = kp[1] * images.shape[-2]
                            draw.ellipse([tuple(kp[:2]-3),tuple(kp[:2]+3)], outline=tuple([255,0,0]),width=2)
                        img.save(viz_folder_hq + img_name+"_gt.png")
                    counter = counter+1

if __name__ == '__main__':
    main()
    