import torchvision
import torch
from utils.tps import get_tps_para,get_thin_plate_spline_grid

def tps_transform(images_org, scal, scal_var, tps_scal, off_scal, rot_scal, masks=None):
    B, _,H,W = images_org.shape
    
    # generate tps params
    coord, t_vector = get_tps_para(images_org.shape[0],scal, scal_var, tps_scal, off_scal, rot_scal)
    
    # generate tps grid
    tps_grid = get_thin_plate_spline_grid(coord, t_vector, (images_org.shape[-2],images_org.shape[-1]))
    
    # apply tps on image
    images = torch.nn.functional.grid_sample(images_org, tps_grid,align_corners=False)
    
    # apply tps on blank image to generate mask
    if masks is None:
        masks = torch.ones(B,3,H,W)
    masks = torch.nn.functional.grid_sample(masks, tps_grid,align_corners=False)
    masks = torch.where(masks > 0.9, torch.tensor([1.]), torch.tensor([0.]))
    masks = -torch.nn.MaxPool2d(5, stride=1,padding=2)(-masks)
    tps_grid = tps_grid[0]
    
    return images, masks, tps_grid

def gen_image_pairs(images, color_transform = False, scal=0.9, scal_var=0.05, tps_scal=0.05, off_scal=0.05, rot_scal=0.25,padding='Reflection',const = 1):
    B, _,H,W = images.shape
    pad = int(H/5)
    if padding == 'Reflection':
        images_padded = torch.nn.ReflectionPad2d(pad)(images)
    elif padding == 'Constant':
        images_padded = torch.nn.ConstantPad2d(pad,const)(images)
    
    # apply tps
    images_tps_1, masks_tps_1, _ = tps_transform(images_padded, 0.98, 0.02, 0.03, 0.02, 0.05)
    
    # apply tps second time
    images_tps_2, masks_tps_2, tps_grid = tps_transform(images_tps_1, scal, scal_var, tps_scal, off_scal, rot_scal, masks=masks_tps_1)

    # color jitter
    if color_transform:
        images_tps_1 = torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.25)(images_tps_1)
        images_tps_2 = torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.25)(images_tps_2)

    # concat images and masks
    images = torch.cat([images_tps_1[:,:,pad:pad+H,pad:pad+W],images_tps_2[:,:,pad:pad+H,pad:pad+W]],dim=0)
    masks = torch.cat([masks_tps_1[:,:,pad:pad+H,pad:pad+W],masks_tps_2[:,:,pad:pad+H,pad:pad+W]],dim=0)
    
    return images, masks, tps_grid