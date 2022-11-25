# tet_debug_wo_rel_affine_sm_l_equv_l1_adaptive_orth_mse_lrp-01-8_softmin_log_models
# clevr_48_run10_models
# tet_114_run4
import torch
import torchvision
import numpy as np
import os
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from matplotlib import cm
from sklearn.metrics import adjusted_rand_score
from scipy.optimize import linear_sum_assignment

from utils.config import print_config, get_config
from utils.torch_utils import to_gpu
from datasets.dataset import dataset
from models.network import NetWork





def build_composite_image(images,features_image,alpha,rgb,color_map,detection=None,boarder=2):
    B,C,H,W = images.shape
    _,N,_,_,_ = alpha.shape
    if not detection is None:
        canvas = np.ones((B,C,H,(2*N+4)*W+(2*N+4-1)*boarder))
    else:
        canvas = np.ones((B,C,H,(2*N+3)*W+(2*N+3-1)*boarder))
    canvas[:,:,:,:W] = images
    counter = 1
    canvas[:,:,:,W+boarder*counter:(2*W)+boarder*counter] = features_image
    counter = counter +1
    masks = np.argmax(alpha, axis=1)
    idx = np.stack((np.repeat(masks,C,axis=1),np.tile(np.arange(3).reshape((1,3,1,1)),(B,1,H,W))),axis=0)
    mask_image = color_map[tuple(idx)]
    canvas[:,:,:,2*W+boarder*counter:(3*W)+boarder*counter] = mask_image
    counter = counter +1
    for i in range(N):
        canvas[:,:,:,(i+3)*W+boarder*counter:(i+4)*W+boarder*counter] = rgb[:,i,:,:,:]
        counter = counter +1
    for i in range(N):
        canvas[:,:,:,(i+N+3)*W+boarder*counter:(i+4+N)*W+boarder*counter] = alpha[:,i,:,:,:]
        counter = counter +1
    if not detection is None:
        canvas[:,:,:,(N+N+3)*W+boarder*counter:(N+4+N)*W+boarder*counter] = detection
    return canvas

def build_prototype_image(alpha,rgb,color_map):
    B,N,C,H,W = rgb.shape
    canvas = np.zeros((B,C,H,(2*N)*W))
    for i in range(N):
        canvas[:,:,:,(i*2)*W:(i*2+1)*W] = rgb[:,i,:,:,:]
    for i in range(N):
        canvas[:,:,:,(i*2+1)*W:(i*2+2)*W] = alpha[:,i,:,:,:]
    return canvas

def add_keypoints_conf(self, images, keypoints,label,min_d = 2):
    keypoints[:,:,2] = keypoints[:,:,2]/torch.max(keypoints[:,:,2],dim=-1)[0].reshape([keypoints.shape[0],1])*images.shape[-1]/30+min_d
    keypoints = torch.cat((keypoints,keypoints[:,:,[2]]), dim=-1)
    keypoints[:,:,:2] = keypoints[:,:,:2] - keypoints[:,:,2:]/2
    keypoints[:,:,2:] = keypoints[:,:,:2] + keypoints[:,:,2:]
    image_np = np.zeros(images.shape, dtype=np.uint8)
    for i in range(images.shape[0]):
        img = images[i,...].cpu().permute(1, 2, 0).numpy()
        H,W,_ = img.shape
        img = Image.fromarray(img, 'RGB')
        draw = ImageDraw.Draw(img)
        for j in range(keypoints.shape[1]):
            kp = keypoints[i,j,:].tolist()
            if label == None:
                draw.ellipse(kp, outline=(0,255,0),width=2)
            else:
                draw.ellipse(kp, outline=tuple(self.color_map[label[i,j]]),width=2)

        img = np.asarray(img)

        image_np[i,...] = np.transpose(img, (2, 0, 1))

    return torch.from_numpy(image_np)

def detection_viz(images, keypoints):
    images = (images * 255.0).astype(np.uint8)
    keypoints[:,:,2] = keypoints[:,:,2]*images.shape[-1]/10
    keypoints = np.concatenate((keypoints,keypoints[:,:,[2]]), axis=-1)
    keypoints[:,:,:2] = keypoints[:,:,:2] - keypoints[:,:,2:]/2
    keypoints[:,:,2:] = keypoints[:,:,:2] + keypoints[:,:,2:]

    image_np = np.zeros(images.shape, dtype=np.uint8)
    for i in range(images.shape[0]):
        img = np.transpose(images[i], (1,2,0))
        H,W,_ = img.shape
        img = Image.fromarray(img, 'RGB')
        draw = ImageDraw.Draw(img)
        for j in range(keypoints.shape[1]):
            kp = keypoints[i,j,:].tolist()
            draw.ellipse(kp, outline=(0,255,0),width=2)

        img = np.asarray(img)
        image_np[i,...] = np.transpose(img, (2, 0, 1))


    return image_np.astype(np.float)/255.0

def average_ari(alpha, instances, foreground_only=False):
    ari = []
    for i, m in enumerate(alpha):

        masks_pred = np.argmax(m.squeeze().numpy(), axis=0).flatten()
        masks_gt = instances[i].numpy().flatten()
        if foreground_only:
            masks_pred = masks_pred[np.where(masks_gt > 0)]
            masks_gt = masks_gt[np.where(masks_gt > 0)]
        score = adjusted_rand_score(masks_pred, masks_gt)
        ari.append(score)
    return sum(ari)/len(ari), ari

def nms(kps,alpha):

    bbox = torch.zeros(kps.shape[0],4)
    bbox[:,0] = kps[:,0] - 5
    bbox[:,1] = kps[:,1] - 5
    bbox[:,2] = kps[:,0] + 5
    bbox[:,3] = kps[:,1] + 5
    score = kps[:,2]
    nms_idx = torchvision.ops.nms(bbox,score,0.3)
    kps = kps[nms_idx]
    labels = labels[nms_idx]
    return kps,labels

def main():
    config = get_config()

    config.training.loss.feature.eqvar.activate = False
    config.training.batch_size = 30
    print_config(config)
    mode = config.mode

    color_map = cm.get_cmap('tab10', 10)
    color_map = color_map(np.linspace(0, 1, 10))
    color_map = (color_map[:,:3]*255).astype(int)
    viz_folder_hq = '{}/{}/{}/{}/viz/'.format(config.metadata.result_root_folder,config.data.dataset.name,config.metadata.name,mode)
    if not os.path.isdir(viz_folder_hq):
        os.makedirs(viz_folder_hq)

    # init data loader
    dataset_tr, _ = dataset(config, mode=mode,evaluate=True)
    # init network
    network = NetWork(config.model)

    # load model
    print('Loading saved models: {}'.format(config.metadata.model_dir + '/' + config.metadata.name+'_models'))
    checkpoint = torch.load(os.path.join(config.metadata.model_dir, config.metadata.name + '_models'))
    network.load_state_dict_ddp(checkpoint['model'])
    network = network.cuda()
    ari_list = []

    if config.data.dataset.sub_name == 'tetrominoes':
        shape_mat = np.zeros((config.model.num_cluster,19))
        color_mat = np.zeros((config.model.num_cluster,6))
        all_mat = np.zeros((config.model.num_cluster,114))
    elif config.data.dataset.sub_name == 'CLEVR':
        shape_mat = np.zeros((config.model.num_cluster,3))
        color_mat = np.zeros((config.model.num_cluster,8))
        size_mat = np.zeros((config.model.num_cluster,2))
        all_mat = np.zeros((config.model.num_cluster,48))  
    
    plot_prototypes = True
    with torch.no_grad():
        for idx, x in tqdm(enumerate(dataset_tr), smoothing=0.1):
            # get data
            images, gt_masks, _, _, _, shape_labels, color_labels, size_labels, sc_labels, img_names = to_gpu(x,0)
            
            B,_,_,_ = images.shape

            # sanity check on data format
            if len(images.shape)!=4 or images.shape[1]!=3:
                raise RunTimeError('Images does not have dimension (B,3,H,W)')
    
            

            # run network
            features_image, keypoints, labels, _, _, diag = network.forward(images, opt_prototypes = False)
            _,K,_ = keypoints.shape

            # calculate ari
            alpha = diag['alpha']
            ari_fg, _ = average_ari(alpha.detach().cpu(), gt_masks.cpu(), foreground_only=True)
            ari_list.append(ari_fg)

            alpha_mask = diag['alpha'].cpu().detach() > 0.8
            for i in range(B):
                for j in range(K):
                    alpha_labels = gt_masks[i][alpha_mask[i,j,0]]
                    kp_idx = -1
                    for k in range(K):
                        if torch.sum(alpha_labels == k+2)/alpha_labels.shape[0] >0.5:
                            kp_idx = k
                    if kp_idx != -1:
                        shape_mat[labels[i,j,0],shape_labels[i][kp_idx]] += 1
                        color_mat[labels[i,j,0],color_labels[i][kp_idx]] += 1
                        all_mat[labels[i,j,0],sc_labels[i][kp_idx]] += 1
                        if config.data.dataset.sub_name == 'CLEVR':
                            size_mat[labels[i,j,0],size_labels[i][kp_idx]] += 1
                                

            if idx < 2:
                detection_img = detection_viz(images.cpu().detach().numpy(), keypoints.cpu().detach().numpy())   
                composite_imgs = build_composite_image(detection_img,features_image.cpu().detach().numpy(),diag['alpha'].cpu().detach().numpy(),diag['rgb'].cpu().detach().numpy(),color_map)
                for _composite_img, _img_name in zip(composite_imgs,img_names):
                    _composite_img = (np.transpose(_composite_img, (1, 2, 0))*255).astype('uint8')
                    _composite_img = Image.fromarray(_composite_img, 'RGB')
                    _composite_img.save(viz_folder_hq + _img_name+".png")

        shape_acc = np.sum(np.max(shape_mat,axis=-1))/np.sum(shape_mat)
        color_acc = np.sum(np.max(color_mat,axis=-1))/np.sum(color_mat)
        
        cost_mat = 1-all_mat/(np.sum(all_mat,axis=0)+1e-5)
        row_ind, col_ind = linear_sum_assignment(cost_mat)
        cost_mat[row_ind, col_ind].sum()
        print('shape acc: {}  color acc: {}'.format(shape_acc,color_acc))
        if config.data.dataset.sub_name == 'CLEVR':
            size_acc = np.sum(np.max(size_mat,axis=-1))/np.sum(size_mat)
            print('size acc: {}'.format(size_acc))
        avg_ari = sum(ari_list) / len(ari_list)
        print ('ari: {}'.format(avg_ari))

if __name__ == '__main__':
    main()
    