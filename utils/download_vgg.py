import torchvision
import torch
import os
path = '../pretrained'
os.makedirs(path, exist_ok=True) 
vgg16 = torchvision.models.vgg16(pretrained=True)
torch.save(vgg16.state_dict(), os.path.join(path,'vgg16'))