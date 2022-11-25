import torchvision
import torch

class kmean_loss(torch.nn.Module):
    def __init__(self,config):
        super(kmean_loss, self).__init__()
        pass

    def forward(self, p, debug=False, inverse=False):
        N,_ = p.shape
        l = torch.argmax(p,dim=1)
        cluster_loss = -torch.mean(torch.log(p[torch.arange(N),l]))
        return cluster_loss, {}

class mse_loss(torch.nn.Module):
    def __init__(self, config):
        super(mse_loss, self).__init__()

    def forward(self, im_recon, im_gt, mask=None):
        loss = torch.nn.MSELoss()(im_recon, im_gt) 
        return loss, {}

class perceptual_loss(torch.nn.Module):
    def __init__(self,pretrained_path):
        super(perceptual_loss, self).__init__()
        vgg16 = torchvision.models.vgg16(pretrained=False)
        vgg16.load_state_dict(torch.load(pretrained_path))
        self.vgg_blocks = torch.nn.ModuleList()
        self.vgg_blocks.append(torch.nn.Sequential(*list(vgg16.features[:4].eval())))
        self.vgg_blocks.append(torch.nn.Sequential(*list(vgg16.features[4:9].eval())))
        self.vgg_blocks.append(torch.nn.Sequential(*list(vgg16.features[9:16].eval())))
        self.vgg_blocks.append(torch.nn.Sequential(*list(vgg16.features[16:23].eval())))
        for _vgg_block in self.vgg_blocks:
            for p in _vgg_block.parameters(): p.requires_grad = False
        self.transform = torch.nn.functional.interpolate
        self.register_buffer('mean',torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std',torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, im_recon, im_gt, mask=None):
        im_recon = (im_recon-self.mean) / self.std
        im_gt = (im_gt-self.mean) / self.std
        stride = 224/max(im_recon.shape[-1],im_recon.shape[-2])

        x = torch.nn.functional.interpolate(im_recon, mode='bilinear', size=(int(im_recon.shape[-2]*stride), int(im_recon.shape[-1]*stride)), align_corners=False)
        y = torch.nn.functional.interpolate(im_gt, mode='bilinear', size=(int(im_recon.shape[-2]*stride), int(im_recon.shape[-1]*stride)), align_corners=False)
        loss = 0.0

        for _vgg_block in self.vgg_blocks:
            x = _vgg_block(x)
            y = _vgg_block(y)
            loss += torch.nn.functional.l1_loss(x, y)

        return loss, {}

class sliced_wasserstein_loss(torch.nn.Module):
    def __init__(self,config):
        super(sliced_wasserstein_loss, self).__init__()
        self.num_projection = 100
        
    def get_cluster_ratio(self, prototypes,features,p=2):
        K,C = prototypes.shape
        dist_mat = torch.sum(torch.pow((prototypes.unsqueeze(0)  - features.unsqueeze(1)),2),dim=-1)
        min_idxs = torch.argmin(dist_mat, dim=1)
        cluster_ratio = torch.zeros(K)
        prototype_var = torch.zeros(K)
        for i in range(K):
            cluster_ratio[i] = torch.sum(min_idxs==i)
            if cluster_ratio[i] !=0:
                prototype_var[i] = torch.mean(pow(features[min_idxs==i] - prototypes[i],2))
            else:
                prototype_var[i] = 1
        var_var = torch.mean(torch.pow((prototype_var - torch.mean(prototype_var)),2))
        cluster_ratio = cluster_ratio + torch.sum(cluster_ratio)*(0.01+var_var)
        cluster_ratio = cluster_ratio/torch.sum(cluster_ratio)
        return cluster_ratio

    def get_sampled_prototypes(self, prototypes, features,rank):
        K,C = prototypes.shape
        N,C = features.shape
        cluster_ratio = self.get_cluster_ratio(prototypes,features,p=2)
        cluster_ratio = (cluster_ratio*features.shape[0]).long()
        cluster_ratio[-1] += features.shape[0]-torch.sum(cluster_ratio)
        # sample prototype based on cluster ratio
        sampled_prototypes = torch.zeros(0,C).to(rank)
        for idx, _cluster_ratio in enumerate(cluster_ratio):
            sampled_prototypes = torch.cat([sampled_prototypes, prototypes[[idx]].repeat([_cluster_ratio,1])],dim=0)
        # add noise to prototype
        sampled_prototypes = sampled_prototypes+ torch.randn((N,C)).to(rank)/50
        sampled_prototypes = sampled_prototypes/torch.norm(sampled_prototypes,dim=1,keepdim=True)
        return sampled_prototypes

    def forward(self, prototypes, features, rank):
        assert len(features.shape)==3, 'features should have dimension of (B,N,C)'
        assert len(prototypes.shape)==2, 'prototypes should have dimension of (K,C)'
        B,N,C_f = features.shape
        K,C_p = prototypes.shape
        assert C_f == C_p, 'prototypes and prototype should have same number of channels'
        # flatten batch
        features = features.view(-1,C_f)
        # sample prototypes
        sampled_prototypes = self.get_sampled_prototypes(prototypes, features,rank)
        # generate random projection
        theta = torch.nn.functional.normalize(torch.randn((C_f, self.num_projection),requires_grad=False), dim=0, p=2).to(rank)
        # project and sort features
        project_features = features@theta
        sorted_features, _ = torch.sort(project_features,0)
        # project and sort embeddings
        project_prototypes = sampled_prototypes@theta
        sorted_prototypes, _ = torch.sort(project_prototypes,0)
        # L2 wasserstein distance
        w_dist = torch.sum(torch.pow((sorted_prototypes-sorted_features),2),dim=1)
        # average distance
        errG = torch.mean(w_dist)
        
        return errG
        