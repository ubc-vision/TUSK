import torch
import torchvision
import os
# modified based on https://github.com/usuyama/pytorch-unet
def double_conv(in_channels, out_channels, padding=1, group_norm=False, group_norm_channel=32):
    if group_norm:
        return torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, out_channels, 3, padding=padding),
                    torch.nn.GroupNorm(group_norm_channel, out_channels),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(out_channels, out_channels, 3, padding=padding),
                    torch.nn.GroupNorm(group_norm_channel, out_channels),
                    torch.nn.ReLU(inplace=True)
                    )     
    else:
        return torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, out_channels, 3, padding=padding),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(out_channels, out_channels, 3, padding=padding),
                    torch.nn.ReLU(inplace=True)
                    )     

def single_conv(in_channels, out_channels, padding=1, group_norm=False):
    if group_norm:
        return torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, out_channels, 3, padding=padding),
                    torch.nn.GroupNorm(32, out_channels),
                    torch.nn.ReLU(inplace=True)
                    )     
    else:
        return torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, out_channels, 3, padding=padding),
                    torch.nn.ReLU(inplace=True)
                    )     

def d_conv(in_channels, out_channels, padding=1, group_norm=False):
    if group_norm:
        return torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, out_channels, 3, padding=padding),
                    torch.nn.GroupNorm(32, out_channels),
                    torch.nn.ReLU(inplace=True)
                    )     
    else:
        return torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, out_channels, 3, padding=padding),
                    torch.nn.ReLU(inplace=True)
                    )     


class UNet(torch.nn.Module):
    def __init__(self, in_channel, out_channel, min_channel, upsample=True, group_norm=True, group_norm_channel=-1):
        super(UNet, self).__init__()
        if group_norm_channel==-1:
            group_norm_channel = min_channel
        self.dconv_down1 = double_conv(in_channel, min_channel, group_norm = group_norm, group_norm_channel=group_norm_channel)
        self.dconv_down2 = double_conv(min_channel, min_channel*2, group_norm = group_norm, group_norm_channel=group_norm_channel)
        self.dconv_down3 = double_conv(min_channel*2, min_channel*4, group_norm = group_norm, group_norm_channel=group_norm_channel)
        self.dconv_down4 = double_conv(min_channel*4, min_channel*8, group_norm = group_norm, group_norm_channel=group_norm_channel)        

        self.maxpool = torch.nn.MaxPool2d(2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)   

        self.dconv_up3 = double_conv(min_channel*4 + min_channel*8, min_channel*4, group_norm = group_norm, group_norm_channel=group_norm_channel)
        self.dconv_up2 = double_conv(min_channel*2 + min_channel*4, min_channel*2, group_norm = group_norm, group_norm_channel=group_norm_channel)
        self.dconv_up1 = double_conv(min_channel + min_channel*2, min_channel, group_norm = group_norm, group_norm_channel=group_norm_channel)

        self.conv_last = torch.nn.Conv2d(min_channel, out_channel, 1)
        # self.feature_head = double_conv(min_channel, out_channel, group_norm = group_norm)
        # self.heatmap_head_1 = double_conv(min_channel, min_channel, group_norm = group_norm)
        # self.heatmap_head_2 = torch.nn.Conv2d(min_channel, 1, 1)
        # self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)    
  
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        x = self.conv_last(x)

        return x

class UNetDecoder(torch.nn.Module):
    def __init__(self, in_channel, out_channel, decoder_channel, group_norm_channel=-1):
        super(UNetDecoder, self).__init__()
        if group_norm_channel==-1:
            group_norm_channel = decoder_channel
        self.decoder = UNet(in_channel,out_channel, decoder_channel, group_norm=True, group_norm_channel=group_norm_channel)
        self.sigmoid = torch.nn.Sigmoid()
        self.alpha = out_channel==4
    def forward(self, x):

        x = self.decoder(x)
        if not self.alpha:
            x = self.sigmoid(x)
        else:
            x_rgb = self.sigmoid(x[:,:3,:,:])
            x_alpha = x[:,[3],:,:]
            x =torch.cat([x_rgb,x_alpha],axis=1)
        return x


# class UNet_Encoder(torch.nn.Module):
#     def __init__(self, latent_channel,encoder_channel):
#         super(UNet_Encoder, self).__init__()
#         self.encoder_features = UNet(3, latent_channel, encoder_channel, group_norm=True)
#         self.encoder_heatmaps = UNet(3, 1, encoder_channel, group_norm=True)

#     def forward(self, x):
#         f = self.encoder_features(x)
#         h = self.encoder_heatmaps(x)
#         return f, h, {}

class UNet_Encoder(torch.nn.Module):
    def __init__(self, in_channel, out_channel,encoder_channel,affine):
        super(UNet_Encoder, self).__init__()
        self.encoder_features = UNet(in_channel, out_channel, encoder_channel, group_norm=True)
        self.heatmap_header = torch.nn.Sequential(
                    torch.nn.Conv2d(out_channel, out_channel, 3, padding=1),
                    torch.nn.GroupNorm(out_channel, out_channel),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv2d(out_channel, 1, 3, padding=1),
                    torch.nn.GroupNorm(1, 1,affine=affine),
                    # torch.nn.ReLU(inplace=True)
                    )    
        # self.encoder_heatmaps = UNet(3, 1, encoder_channel, group_norm=True)

    def forward(self, x):
        f = self.encoder_features(x)
        h = self.heatmap_header(f)
        return f, h, {}

