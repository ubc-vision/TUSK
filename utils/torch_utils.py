import torch
import math

def to_gpu(x,rank):
    if isinstance(x, torch.Tensor):
        return x.to(rank)
    list_cuda = []
    for t in x:
        if isinstance(t, torch.Tensor):
            list_cuda.append(t.to(rank))
        else:
            list_cuda.append(t)
    return tuple(list_cuda)

def spatial_softmax(heatmaps, kernel_size, strength=2):

    # heatmap [N, C, H, W]
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    pad = kernel_size // 2

    max_logits = torch.max(heatmaps, 2, True)[0]
    max_logits = torch.max(max_logits, 3, True)[0]

    ex = torch.exp(strength * (heatmaps - max_logits))
    # ex = torch.exp(strength * (heatmap))
    sum_ex = torch.nn.functional.avg_pool2d(ex, kernel_size=kernel_size, stride=1, count_include_pad=False) * kernel_size**2
    sum_ex = torch.nn.functional.pad(sum_ex, pad=(pad, pad, pad, pad), mode='replicate')
    probs = ex / (sum_ex + 1e-6)
    return probs

class SamplePointsDiff(torch.nn.Module):
    def __init__(self,kernel_size, topk):
        super(SamplePointsDiff, self).__init__()

        assert kernel_size%2==1, "kernel size for point sampler must be an odd number!"
        self.kernel_size = kernel_size
        self.padding = kernel_size//2
        self.register_buffer('x_kernel', (torch.arange(kernel_size)-kernel_size//2).view(1,1,1,-1).repeat(1,1,kernel_size,1).type(torch.FloatTensor))
        self.register_buffer('y_kernel', (torch.arange(kernel_size)-kernel_size//2).view(1,1,-1,1).repeat(1,1,1,kernel_size).type(torch.FloatTensor))
        self.register_buffer('s_kernel', torch.ones(kernel_size,kernel_size).view(1,1,kernel_size,kernel_size))
        self.maxpool = torch.nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size//2)
        self.topk = topk

    def forward(self, score_map):
        # print('Using old sample points')
        cur_dev = score_map.device
        diag = {}
        B, _, H, W = score_map.shape
        # index map (B,1,H,W,2)
        x = torch.arange(0, W, 1,device=cur_dev).unsqueeze(0).repeat(H,1)
        y = torch.arange(0, H, 1,device=cur_dev).unsqueeze(-1).repeat(1,W)
        xy = torch.stack([x,y],dim=-1).view(1,H,W,2).repeat(B,1,1,1)
        
        # nms
        output = self.maxpool(score_map)
        mask = torch.eq(output, score_map)
        filtered_score_map = output*mask
        
        # extract points
        _, topk_idx = torch.topk(filtered_score_map.view(B,-1),self.topk)
        keypoints = xy.view(B,-1,2)[torch.arange(B).unsqueeze(-1).repeat(1,self.topk),topk_idx,:]

        zero_padding = torch.nn.ZeroPad2d(self.padding)
        score_map_padded = zero_padding(score_map)
        score_map_padded = score_map_padded
        score_map_padded = score_map_padded/0.2
        x_offset = torch.nn.functional.conv2d(torch.exp(score_map_padded), self.x_kernel)
        y_offset = torch.nn.functional.conv2d(torch.exp(score_map_padded), self.y_kernel)
        norm_const =  torch.nn.functional.conv2d(torch.exp(score_map_padded), self.s_kernel)

        x_offset = x_offset/norm_const
        y_offset = y_offset/norm_const
        keypoints = xy.view(B,-1,2)[torch.arange(B).unsqueeze(-1).repeat(1,self.topk),topk_idx,:]
        keypoints_x_offset = x_offset.view(B,-1)[torch.arange(B).unsqueeze(-1).repeat(1,self.topk),topk_idx]
        keypoints_y_offset = y_offset.view(B,-1)[torch.arange(B).unsqueeze(-1).repeat(1,self.topk),topk_idx]
        keypoints_offset = torch.stack([keypoints_x_offset,keypoints_y_offset],dim=-1)
        keypoints = keypoints + keypoints_offset


        # confidence
        # bilinear interpolate

        keypoints_00 = torch.clamp(torch.cat([torch.floor(keypoints[:,:,[0]]),torch.floor(keypoints[:,:,[1]])],dim=2),0,H-1)
        keypoints_10 = torch.clamp(torch.cat([torch.ceil(keypoints[:,:,[0]]),torch.floor(keypoints[:,:,[1]])],dim=2),0,H-1)
        keypoints_01 = torch.clamp(torch.cat([torch.floor(keypoints[:,:,[0]]),torch.ceil(keypoints[:,:,[1]])],dim=2),0,H-1)
        keypoints_11 = torch.clamp(torch.cat([torch.ceil(keypoints[:,:,[0]]),torch.ceil(keypoints[:,:,[1]])],dim=2),0,H-1)
        idx = torch.cat((torch.arange(B,device=cur_dev).reshape(-1,1,1,1).repeat(1,self.topk,4,1),
                        torch.zeros(B,self.topk,4,1,device=cur_dev),
                        torch.unsqueeze(torch.cat([keypoints_00[:,:,[1]],keypoints_10[:,:,[1]],keypoints_01[:,:,[1]],keypoints_11[:,:,[1]]],dim=-1),-1),
                        torch.unsqueeze(torch.cat([keypoints_00[:,:,[0]],keypoints_10[:,:,[0]],keypoints_01[:,:,[0]],keypoints_11[:,:,[0]]],dim=-1),-1))
                        ,dim=-1).type(torch.long).permute(3,0,1,2)
        keypoints_conf = score_map[tuple(idx)]

        dist_x = torch.stack((keypoints_11[:,:,0]-keypoints[:,:,0],keypoints[:,:,0]-keypoints_01[:,:,0],keypoints_10[:,:,0]-keypoints[:,:,0],keypoints[:,:,0]-keypoints_00[:,:,0]),dim=-1)
        dist_y = torch.stack((keypoints_11[:,:,1]-keypoints[:,:,1],keypoints_01[:,:,1]-keypoints[:,:,1],keypoints[:,:,1]-keypoints_10[:,:,1],keypoints[:,:,1]-keypoints_00[:,:,1]),dim=-1)
        dist_weight = dist_x*dist_y

        keypoints_conf = torch.sum(keypoints_conf*dist_weight,dim=-1)

        diag['keypoints_conf'] = keypoints_conf
        diag['filtered_score_map'] = filtered_score_map

        keypoints = torch.cat((keypoints,keypoints_conf.unsqueeze(-1)),dim=-1)

        return keypoints, diag

def sample_features(featuremap,keypoints):
    cur_dev = featuremap.device
    B, C, H, W = featuremap.shape
    _,N,_ = keypoints.shape
    idx = torch.cat((torch.arange(B,device=cur_dev).reshape(-1,1,1,1).repeat(1,N,C,1),
                     torch.arange(C,device=cur_dev).reshape(1,1,-1,1).repeat(B,N,1,1),
                     keypoints[:,:,[1]].unsqueeze(2).repeat(1,1,C,1),
                     keypoints[:,:,[0]].unsqueeze(2).repeat(1,1,C,1)),dim=-1).type(torch.long).permute(3,0,1,2)
    features = featuremap[tuple(idx)]
    return features


def reconstruct_featuremap(keypoints,features,outshape):
    cur_dev = featuremap.device
    B, C, H, W = outshape
    _,N,_ = keypoints.shape
    featuremap = torch.zeros(outshape)
    idx = torch.cat((torch.arange(B,device=cur_dev).reshape(-1,1,1,1).repeat(1,N,C,1),
                     torch.arange(C,device=cur_dev).reshape(1,1,-1,1).repeat(B,N,1,1),
                     keypoints[:,:,[1]].unsqueeze(2).repeat(1,1,C,1),
                     keypoints[:,:,[0]].unsqueeze(2).repeat(1,1,C,1)),dim=-1).type(torch.long).permute(3,0,1,2)

    featuremap[tuple(idx)] = features * keypoints[:,:,[2]]
    return featuremap  
                    
def gen_gaussian_weights(var,kernel_size=None, H=None, W=None):
    if kernel_size!=None:
        H = kernel_size
        W = kernel_size
    # calculted discretelly
    weights = torch.zeros(H,W)
    center_y = H//2
    center_x = W//2
    # norm_factor = (2*math.pi*var)
    for x in range(W):
        for y in range(H):
            delta_x = x - center_x
            delta_y = y - center_y
            weights[y,x] = math.exp(-0.5*(delta_x**2+delta_y**2)/var)
    # norm_factor_disc = torch.sum(weights)
    # assert abs(norm_factor-norm_factor_disc)/(norm_factor+norm_factor_disc) <0.2, 'Gaussian kernel weights have a larger numericle error'
    norm_factor = torch.max(weights)
    weights = weights/norm_factor
    
    return weights

def gen_dog_weights(var,kernel_size,norm):


    weights_x = torch.zeros(kernel_size,kernel_size)
    weights_y = torch.zeros(kernel_size,kernel_size)
    center = kernel_size//2
    for x in range(kernel_size):
        for y in range(kernel_size):
            delta_x = x - center
            delta_y = y - center
            weights_x[y,x] = math.exp(-0.5*(delta_x**2+delta_y**2)/var)*(-delta_x)
            weights_y[y,x] = math.exp(-0.5*(delta_x**2+delta_y**2)/var)*(-delta_y)
    norm_factor_x = torch.max(weights_x)
    norm_factor_y = torch.max(weights_y)
    weights_x = weights_x/norm_factor_x
    weights_y = weights_y/norm_factor_y
    if norm:
        weights_x = (weights_x+1)/2
        weights_y = (weights_y+1)/2
 
    return weights_x, weights_y

class GaussianRecon(torch.nn.Module):
    def __init__(self,var,norm=True):
        super(GaussianRecon, self).__init__()
        self.var = var
        self.norm = norm
    def forward(self,keypoints,output_shape):
        cur_dev = keypoints.device
        H, W = output_shape
        B, N, _ = keypoints.shape
        grid_x = torch.arange(W,device=cur_dev).reshape(1,1,1,-1).repeat(B,N,H,1)
        grid_y = torch.arange(H,device=cur_dev).reshape(1,1,-1,1).repeat(B,N,1,W)
        grid = torch.stack([grid_x,grid_y],dim=-1)
        offset = grid - keypoints[:,:,:2].reshape(B,N,1,1,2)
        exp_term = -torch.sum(offset*offset/self.var,dim=-1)*0.5
        heatmap = torch.exp(exp_term)
        if self.norm:
            heatmap = heatmap/torch.sum(heatmap.view(B,N,-1),dim=-1).view(B,N,1,1)

        return heatmap
