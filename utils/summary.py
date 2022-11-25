import torch
import torchvision
import numpy as np
from tensorboardX import SummaryWriter
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib
from io import BytesIO
from matplotlib import cm
from sklearn.manifold import TSNE
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

def figure_to_numpy(figure, close=True):
    buf = BytesIO()
    figure.savefig(buf, format='png', bbox_inches='tight', transparent=False, dpi=300)
    if close:
        plt.close(figure)
    buf.seek(0)
    arr = matplotlib.image.imread(buf, format='png')[:,:,:3]
    arr = np.moveaxis(arr, source=2, destination=0)
    return arr

class CustomSummaryWriter(SummaryWriter):
    def __init__(self, log_dir, num_cluster):
        super(CustomSummaryWriter, self).__init__(log_dir)
        # if config.num_cluster <= 10: 
        #     color_map = cm.get_cmap('tab10', 10)
        #     color_map = color_map(np.linspace(0, 1, 10))
        if num_cluster <= 20:  
            color_map = cm.get_cmap('tab20', 20)
            color_map = color_map(np.linspace(0, 1, 20))
        else:
            color_map = cm.get_cmap('gist_rainbow')
            color_map = color_map(np.linspace(0, 1, num_cluster))
        self.color_map_float = color_map[:,:3]
        self.color_map = (color_map[:,:3]*255).astype(int)

    def add_tsne(self, name, embeddings,labels,cluster_center, iteration):
        # import IPython
        # IPython.embed()
        # assert(0)

        _,_,C = embeddings.shape
        if not cluster_center is None:
            N,_ = cluster_center.shape
            x = np.concatenate((cluster_center.detach().cpu().numpy(),embeddings.reshape(-1,C).detach().cpu().numpy()),axis=0)
            y = np.concatenate((np.arange(N),labels.reshape(-1).detach().cpu().numpy()),axis=0)
        else:
            x = embeddings.reshape(-1,C).detach().cpu().numpy()
            y = labels.reshape(-1).detach().cpu().numpy()

        X_embedded = TSNE(n_components=2).fit_transform(x)

        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.gca()
        ax.scatter(X_embedded[:,0],X_embedded[:,1],c= [self.color_map_float[i] for i in y], marker='o',s=10)
        if not cluster_center is None:
            ax.scatter(X_embedded[:N,0],X_embedded[:N,1],c= (0,0,0), marker='x',s=200,linewidths=4)
        # ax.scatter(X_embedded[:N,0],X_embedded[:N,1],c= [self.color_map_float[i] for i in y[:N]], marker='x',s=200,linewidths=2)
        ax.set_xticks([]) # values
        ax.set_xticklabels([]) # labels
        ax.set_yticks([]) # values
        ax.set_yticklabels([]) # labels
        fig.tight_layout()

        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3) 
        img = np.transpose(img,(2,0,1))
        img = torch.from_numpy(img)

        self.add_images(name,img,iteration) 

    def add_images_heatmap(self, name, images, heatmap, iteration):
        heatmap = self.draw_heatmap_on_images(images, heatmap)
        self.add_images(name, heatmap, iteration)

    def add_images_with_label(self, name, images, label, iteration):
        B, C, H, W = images.shape
        images_min = torch.min(images.view(B,C,-1),dim=-1)[0].view(B,C,1,1)
        images_max = torch.max(images.view(B,C,-1),dim=-1)[0].view(B,C,1,1)
        images = ((images - images_min) / (images_max- images_min) * 255.0).byte()
        fnt = ImageFont.load_default()
        image_np = np.zeros(images.shape, dtype=np.uint8)
        for i in range(images.shape[0]):
            img = images[i,...].cpu().permute(1, 2, 0).numpy()
            color = tuple(self.color_map[label[i]])
            img = Image.fromarray(img, 'RGB')
            draw = ImageDraw.Draw(img)
            draw.rectangle([0,0,W-1,H-1], outline=color, fill=None)
            draw.rectangle([1,1,W-2,H-2], outline=color, fill=None)
            draw.rectangle([2,2,W-3,H-3], outline=color, fill=None)
            img = np.asarray(img)
            image_np[i,...] = np.transpose(img, (2, 0, 1))
        self.add_images(name,torch.from_numpy(image_np),iteration)

    def add_images(self, name, images, iteration, mode=None, resize=None, keypoints=None, label=None,images_max=None,images_min=None,grid_size=None, min_d=2):
        # images [B, C, H, W]
        max_images = min(images.shape[0],10)
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        if len(images.shape) == 4:
            images = images[0:max_images, ...]
        elif len(images.shape) == 5:
            images = images[0:max_images, ...]
            images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        else:
            raise Exception('images.shape() {}'.format(images.shape))
        if images_max is not None:
            images_max = images_max[0:max_images, ...]
        if images_min is not None:
            images_min = images_min[0:max_images, ...]

        if resize is not None:
            w = int(images.shape[-1] * resize)
            h = int(images.shape[-2] * resize)
            images = torch.nn.functional.interpolate(images, (h,w), mode='bilinear')
            if keypoints is not None:
                keypoints[:,:,:2] = keypoints[:,:,:2] * resize
            if grid_size is not None:
                grid_size = grid_size * resize
        norm_overwrite = False
        if (images.shape[1]<3):
            images = torch.mean(images,dim=1).unsqueeze(1).repeat(1,3,1,1)
        elif (images.shape[1]>3):
            norm_overwrite = True
            # import IPython
            # IPython.embed()
            # assert(0)
            images = torch.stack([torch.mean(images[:,:10,:,:],dim=1),torch.mean(images[:,10:20,:,:],dim=1),torch.mean(images[:,20:,:,:],dim=1)],dim=1)
            _max = torch.max(images.view(images.shape[0],images.shape[1],-1),dim=-1)[0].view(images.shape[0],images.shape[1],1,1)
            _min = torch.min(images.view(images.shape[0],images.shape[1],-1),dim=-1)[0].view(images.shape[0],images.shape[1],1,1)
            images = ((images-_min)/(_max-_min)* 255.0).byte()
        B, C, H, W = images.shape
        if images_max is None or images_min is None:
            images_min = torch.min(images.view(B,C,-1),dim=-1)[0].view(B,C,1,1)
            images_max = torch.max(images.view(B,C,-1),dim=-1)[0].view(B,C,1,1)
        if not norm_overwrite:
            images = ((images - images_min) / (images_max- images_min) * 255.0).byte()
        # images = (images * 255.0).byte()
        if mode is not None:
            if mode == 'keypoints_conf':
                images = self.add_keypoints_conf(images, keypoints,label,min_d)
        if grid_size is not None:
            images = self.add_grid(images,grid_size)
        image = torchvision.utils.make_grid(images, nrow=max_images, padding=1, pad_value=255)
        self.add_image(name, image, iteration)

    def add_grid(self,images,grid_size):
        image_np = np.zeros(images.shape, dtype=np.uint8)
        for i in range(images.shape[0]):
            img = images[i,...].cpu().permute(1, 2, 0).numpy()
            H,W,_ = img.shape
            img = Image.fromarray(img, 'RGB')
            draw = ImageDraw.Draw(img)
            # add h line
            for j in range(int(H/grid_size-1)):
                draw.line([(0,grid_size*(j+1)),(W,grid_size*(j+1))], fill=(122,255,122),width=2)
            #add v line
            for j in range(int(W/grid_size-1)):
                draw.line([(grid_size*(j+1),0),(grid_size*(j+1),H)], fill=(122,255,122),width=2)
            img = np.asarray(img)

            image_np[i,...] = np.transpose(img, (2, 0, 1))   
        return torch.from_numpy(image_np)

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

    def draw_boxes_on_images(self, images, boxes, labels=None, match=None):
        fnt = ImageFont.load_default()
        image_np = np.zeros(images.shape, dtype=np.uint8)
        for i in range(images.shape[0]):
            img = images[i,...].cpu().permute(1, 2, 0).numpy()
            H,W,_ = img.shape
            img = Image.fromarray(img, 'RGB')
            draw = ImageDraw.Draw(img)
            for j in range(boxes.shape[1]):
                kp = boxes[i,j,:].tolist()
                if match is not None:
                    color = (0,255,0) if int(round(match[i,j,0].item())) else (255,0,0)
                    draw.rectangle(kp, outline=color, fill=None)
                    draw.text((kp[2]-20, kp[3]-20), '{}'.format(int(match[i,j,1].item()*100)), fill=color, font=fnt)
                    if labels is not None:
                        color = (0,255,0) if int(round(match[i,j,2].item())) else (255,0,0)
                        draw.text((kp[0]+2, kp[1]), str(labels[i][j].item()), fill=color, font=fnt)
                        draw.text((W-15*(j+1), H-15), str(labels[i][j].item()), fill=(100,100,0), font=fnt)
                else: 
                    color = (0,0, 255)
                    draw.rectangle(kp, outline=color, fill=None)


            img = np.asarray(img)
            image_np[i,...] = np.transpose(img, (2, 0, 1))
        return torch.from_numpy(image_np)

    def draw_heatmap_on_images(self, images, heatmap):
        plt.switch_backend('agg')
        # plt.set_cmap('jet')
        plt.set_cmap('jet')
        hmin=heatmap.min()
        hmax=heatmap.max()
        fig, axes = plt.subplots(1, images.shape[0], figsize=(16,2))
        if images.shape[0] == 1:
            axes = np.array([axes])
        for i in range(images.shape[0]):
            I = images[i,...].cpu().permute(1, 2, 0).numpy()
            H = heatmap[i,...].detach().cpu().numpy()
            ax = axes[i]
            ax.imshow(I)
            im = ax.imshow(H, alpha=1.0, vmin=hmin, vmax=hmax)

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        fig.subplots_adjust(wspace=0, hspace=0)
        plt.colorbar(im, ax=axes.ravel().tolist())
        output = figure_to_numpy(fig)

        return torch.from_numpy(output)
