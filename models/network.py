import torch 

from models.unet import UNet_Encoder, UNetDecoder
from utils.torch_utils import sample_features, SamplePointsDiff, GaussianRecon

class NetWork(torch.nn.Module):
    def __init__(self, model_config):
        super(NetWork, self).__init__()

        # copy config
        self.model_config = model_config

        # encoder
        self.encoder = UNet_Encoder(3, model_config.latent_channel, model_config.latent_channel,model_config.encoder.gnorm_affine)

        # point sampler
        self.sample_points_diff = SamplePointsDiff(model_config.spatial_softmax.kernel_size, model_config.top_k)

        # heatmap reconstruction
        self.heatmap_recon = GaussianRecon(model_config.gau_var)

        # decoder channel
        decoder_input_channel = model_config.latent_channel

        # decoder output channel
        if not model_config.alpha_blending:
            out_channels = 3
        else:
            out_channels = 4

        # decoder
        self.decoder = UNetDecoder(decoder_input_channel,out_channels,decoder_input_channel)

        # prototype
        self.prototype = torch.nn.Parameter(torch.randn(model_config.num_cluster,model_config.latent_channel)) 
        self.prototype.requires_grad = True

    def load_state_dict_ddp(self,model):
        # check current run mode
        ddp = False

        if list(self.state_dict().keys())[0].split('.')[0] == 'module':
            ddp = True
        if not ddp:
            if list(model.keys())[0].split('.')[0] != 'module':
                self.load_state_dict(model)
            else:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in model.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v
                self.load_state_dict(new_state_dict)           
        else:
            if list(model.keys())[0].split('.')[0] == 'module':
                self.load_state_dict(model)
            else:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in model.items():
                    name = 'module.'+k # add `module.`
                    new_state_dict[name] = v
                self.load_state_dict(new_state_dict)              

    def forward(self, images, opt_prototypes = False):

        B,_,H,W = images.shape

        # encoder to generate featuremaps and heatmaps
        featuremap, heatmap, encoder_diag = self.encoder(images)

        # absolute score map
        score_map = torch.nn.Sigmoid()(heatmap)

        # sample points
        keypoints, sample_points_diag = self.sample_points_diff(score_map)
    
        # extract features
        features = sample_features(featuremap, keypoints)

        # normalize features
        features_norm = torch.nn.functional.normalize(features, dim=-1, p=2)

        # normalize prototype
        prototype_norm = torch.nn.functional.normalize(self.prototype, dim=1, p=2)
        
        # do not need to run decoder for prototype optimization
        if opt_prototypes:
            return features_norm, prototype_norm
        
        _,N,C = features_norm.shape
        K,_ = prototype_norm.shape

        # compute similarity between feature and prototype 
        dist_mat = torch.mean(torch.pow(features_norm.view(B,N,1,C) - prototype_norm.view(1,1,K,C),2),dim=-1)
        p_soft = torch.nn.Softmin(dim=2)(dist_mat*20) # hardcode temperature here
        label = p_soft.max(-1, keepdim=True)[1]

        # reconstruct heatmap
        recon_heatmap = self.heatmap_recon(keypoints, (H,W)) * keypoints[:,:,[2]].unsqueeze(-1)

        # reconstruct featuremap
        features_recon = features_norm.reshape(B,N,C,1,1)*recon_heatmap.unsqueeze(2)

        # decode featuremap
        if not self.model_config.alpha_blending:
            features_image = self.decoder(torch.sum(features_recon,dim=1))
        else:
            features_image = self.decoder(features_recon.view(-1,C,H,W))
            alpha  = features_image.view(-1,N,4,H,W)[:,:,[-1],:,:]
            rgb = features_image.view(-1,N,4,H,W)[:,:,:3,:,:]
            alpha = torch.nn.Softmax(dim=1)(alpha)
            features_image = torch.sum(rgb *alpha,dim=1)

        # add diag info
        network_diag = {}
        network_diag['p_soft'] = p_soft
        network_diag['score_map'] = score_map
        network_diag['featuremap'] = featuremap
        network_diag['features_recon'] = features_recon
        network_diag['filtered_score_map'] = sample_points_diag['filtered_score_map']
        network_diag['heatmap'] = heatmap
        network_diag['recon_heatmap'] = recon_heatmap    
        network_diag['encoder_diag'] = encoder_diag
        if self.model_config.alpha_blending:
            network_diag['rgb'] = rgb
            network_diag['alpha'] = alpha
 
        return features_image, keypoints, label, features_norm, prototype_norm, network_diag
