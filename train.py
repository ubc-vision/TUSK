import os
from tqdm import tqdm

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist


from utils.config import print_config, get_config,config_to_string
from utils.utils import tensorboard_scheduler
from utils.torch_utils import to_gpu
from datasets.dataset import dataset
from models.network import NetWork
from utils.summary import CustomSummaryWriter
from models.loss import mse_loss, perceptual_loss, kmean_loss, sliced_wasserstein_loss

class DataParallelPassthrough(DDP):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class Trainer():
    def __init__(self, config, world_size, rank):
        # copy config
        self.config = config
        self.training_config = config.training
        self.viz_config = config.viz
        self.metadata_config = config.metadata

        # distributed data parallel stuff
        if world_size>1:
            self.ddp = True
        else:
            self.ddp = False
        self.rank = rank
        self.world_size = world_size
        
        # init tensorboard scheduler
        self.tb_scheduler = tensorboard_scheduler(config.training.scheduler)

        # init iteration counter
        self.num_iter = 0
        self.start_epoch = 0
        self.cur_epoch = 0

        # init resume flag
        self.resumed = False
        
        # init best loss with a large number
        self.best_loss = 1000

        # init summary writer
        if rank==0:
            self.summary = CustomSummaryWriter(config.viz.log_dir + '/' + config.metadata.name,config.model.num_cluster) 
        else:
            self.summary = None

        # init autoencoder loss
        # reconstruction loss
        if config.training.loss.feature.recon.type == 'MSE':
            self.recon_loss = mse_loss(config).to(rank)
        elif config.training.loss.feature.recon.type == 'Perceptual':
            self.recon_loss = perceptual_loss(config.training.loss.feature.recon.pretrained_path).to(rank)
        
        # cluster loss
        self.cluster_loss = kmean_loss(config).to(rank)
            
        
        # eqvar loss
        self.eqvar_loss = mse_loss(config).to(rank)

        # sw loss
        self.sw_loss = sliced_wasserstein_loss(config).to(rank)

        # # nan counter
        # self.nan_counter = 0
        
    def write_meta_data(self):
        # add hyper parameters to summary 
        self.summary.add_text('hyper paramter',config_to_string(self.config))
        self.summary.add_text('name',self.metadata_config.name)
        # add comment to summary
        self.summary.add_text('comment',self.metadata_config.comments)

    def resume_model_solver(self, network, network_solver, prototype_solver):
        model_path = os.path.join(self.metadata_config.model_dir, self.metadata_config.name + '_models')
        solver_path = os.path.join(self.metadata_config.model_dir, self.metadata_config.name + '_solvers')

        if os.path.isfile(model_path) and os.path.isfile(solver_path):
            # load network weights
            print('... loading model to cuda {}'.format(self.rank))
            if not self.ddp:
                checkpoint = torch.load(model_path)
                network.load_state_dict_ddp(checkpoint['model'])
                print('model loaded to cuda {}'.format(self.rank))
            else:
                map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
                checkpoint = torch.load(model_path,map_location=map_location)
                network.load_state_dict_ddp(checkpoint['model'])
                print('model loaded to cuda {}'.format(self.rank))
            
            # load solver and other info
            print('... loading saved solvers: {}'.format(solver_path))
            checkpoint = torch.load(solver_path)
            network_solver.load_state_dict(checkpoint['network_solver'])
            if prototype_solver:
                prototype_solver.load_state_dict(checkpoint['prototype_solver'])
            self.num_iter = checkpoint['iteration']
            self.best_loss = checkpoint['best_loss']
            self.start_epoch = checkpoint['epoch'] 
            print('Previous state resumed, continue training at {} iteration {} epoch, best_loss {}'.format(self.num_iter,self.start_epoch,self.best_loss))
            self.resumed = True

        else:
            print('Did not find saved model, fresh start')
            self.resumed = False
        


    def save_model(self, network, network_solver, prototype_solver):

        # save network weights
        print('Saving models: {}'.format(self.metadata_config.model_dir + '/' + self.metadata_config.name+'_models'))
        torch.save({'model': network.state_dict(), 'prototype':network.prototype}, os.path.join(self.metadata_config.model_dir,  self.metadata_config.name+'_models'))
        # save solver and other info
        print('Saving solvers: {}'.format(self.metadata_config.model_dir + '/' + self.metadata_config.name+'_solvers'))
        torch.save({ 
            'network_solver': network_solver.state_dict(),
            'prototype_solver': prototype_solver.state_dict(),
            'iteration': self.num_iter,
            'best_loss': self.best_loss,
            'epoch': self.cur_epoch
            }, os.path.join(self.metadata_config.model_dir, self.metadata_config.name + '_solvers'.format(self.num_iter)))   
              
        print('Model and Solver have been saved')
        


    def train(self, network, dataset_tr, sampler_tr, network_solver, prototype_solver):
        # write training meta data to tensor board
        if self.resumed == False and self.rank==0:
            self.write_meta_data()

        # main training loop
        for epoch in tqdm(range(self.start_epoch, self.training_config.epochs)):
            self.cur_epoch = epoch
            if self.ddp:
                sampler_tr.set_epoch(epoch)
            # train one epoch
            for x in tqdm(dataset_tr, smoothing=0.1):        
                # move data on gpu
                images, _, tps_grid, _, _, _ = to_gpu(x,self.rank)

                if len(images.shape)==4:
                    B,_,H,W = images.shape
                else:
                    images = torch.cat((images[:,0],images[:,1]),dim=0)
                    B,_,H,W = images.shape
                    
                # check batch size
                if B % self.training_config.batch_size!=0 :
                    print('skip partial batch')
                    continue

                ## ---Train encoder and decoder---
                # forward pass
                features_image, keypoints, labels, features_norm, prototype_norm, network_diag = network.forward(images, opt_prototypes = False)
                _, N, C  = features_norm.shape
                                
                # reconstruction loss
                recon_loss_feature, _ = self.recon_loss.forward(features_image, images)
                recon_loss_feature = recon_loss_feature * self.training_config.loss.feature.recon.weight
                
                # cluster loss
                cluster_loss,_ = self.cluster_loss.forward(network_diag['p_soft'].reshape(B*N,-1)) 
                cluster_loss = cluster_loss * self.training_config.loss.feature.cluster.weight

                # equ loss    
                pad = int(H/5)

                # equ loss on scoremap 
                tps_heatmap =  torch.nn.functional.grid_sample(torch.nn.ReflectionPad2d(pad)(network_diag['score_map'][:int(B/2),:,:,:]), tps_grid)[:,:,pad:pad+H,pad:pad+W]
                equ_h_loss, _ = self.eqvar_loss.forward(tps_heatmap,network_diag['score_map'][int(B/2):])
                equ_h_loss = self.training_config.loss.feature.eqvar.h_weight * equ_h_loss
                
                # equ loss on featuremap
                tps_featuremap =  torch.nn.functional.grid_sample(torch.nn.ReflectionPad2d(pad)(network_diag['featuremap'][:int(B/2),:,:,:]), tps_grid)[:,:,pad:pad+H,pad:pad+W]                    
                equ_f_loss, _ = self.eqvar_loss.forward(tps_featuremap,network_diag['featuremap'][int(B/2):])
                equ_f_loss = self.training_config.loss.feature.eqvar.f_weight * equ_f_loss

                # final loss for encoder and decoder
                feature_loss = (cluster_loss+recon_loss_feature+equ_h_loss+equ_f_loss)
                
                #backprop
                network_solver.zero_grad()
                feature_loss.backward()
                network_solver.step()

                ## ---Train embedding---
                for i in range(self.training_config.lr.num_opt_sw):
                    # forward pass
                    features_norm, prototypes_norm = network.forward(images, opt_prototypes = True)

                    # sw loss
                    if self.ddp:
                        # with torch.no_grad():
                        features_all_batch = [torch.zeros(features_norm.shape).to(self.rank) for _ in range(self.world_size)]
                        dist.all_gather(features_all_batch, features_norm)
                        features_all_batch = torch.cat(features_all_batch)
                        # use this to allow gradient flow back to features
                        features_all_batch[self.rank*B:(self.rank+1)*B,:,:] = features_norm
                    
                        labels_all_batch = [torch.zeros(labels.shape).type(torch.long).to(self.rank) for _ in range(self.world_size)]
                        dist.all_gather(labels_all_batch, labels)
                        labels_all_batch = torch.cat(labels_all_batch)

                    else:
                        features_all_batch = features_norm
                        labels_all_batch = labels

                    prototype_loss = self.sw_loss.forward(prototypes_norm, features_all_batch, self.rank)
                    prototype_loss = prototype_loss * self.training_config.loss.prototype.sw.weight

                    # backprop
                    prototype_solver.zero_grad()
                    prototype_loss.backward()
                    prototype_solver.step()  

                ## ---Viz and Evaluation---
                eval_flag, save_flag, valid_flag = self.tb_scheduler.schedule()
                
                if self.rank == 0:
                    with torch.no_grad():
                        # write numbers to tensor board
                        if eval_flag:
                            # loss terms
                            self.summary.add_scalar('01 feature loss', feature_loss, self.num_iter)
                            self.summary.add_scalar('02 reconstruction loss', recon_loss_feature, self.num_iter)
                            self.summary.add_scalar('03 cluster loss', cluster_loss, self.num_iter) 
                            self.summary.add_scalar('04.1 equ h loss', equ_h_loss, self.num_iter) 
                            self.summary.add_scalar('04.2 equ f loss', equ_f_loss, self.num_iter) 
                            self.summary.add_scalar('05 prototype loss', prototype_loss, self.num_iter)
                            # self.summary.add_scalar('10 nan counter', self.nan_counter, self.num_iter)
                        
                        # write images to tensor board
                        if valid_flag:
                            # input images
                            self.summary.add_images('1.1 images tps1', images[:int(B/2)].cpu(), self.num_iter)
                            self.summary.add_images('1.2 images tps2', images[int(B/2):].cpu(), self.num_iter)

                            # reconstruct images
                            self.summary.add_images('2.1 recon images tps1 from features', features_image[:B].cpu(), self.num_iter)
                            self.summary.add_images('2.2 recon images tps2 from features', features_image[int(B/2):].cpu(), self.num_iter)
                            
                            # heatmap raw
                            cat_heatmaps = torch.cat([torch.mean(network_diag['heatmap'][:int(B/2)].cpu(),dim=1).reshape(int(B/2),1,-1),torch.mean(network_diag['heatmap'][int(B/2):].cpu(),dim=1).reshape(int(B/2),1,-1)],dim=-1)
                            heatmaps_raw_min = torch.min(cat_heatmaps,dim=-1)[0].view(int(B/2),1,1,1)
                            heatmaps_raw_max = torch.max(cat_heatmaps,dim=-1)[0].view(int(B/2),1,1,1)
                            self.summary.add_images('3.1 heatmap map', network_diag['heatmap'][:int(B/2)].cpu(), self.num_iter,resize=2,images_min=heatmaps_raw_min,images_max=heatmaps_raw_max)
                            self.summary.add_images('3.2 heatmap on tps image', network_diag['heatmap'][int(B/2):].cpu(), self.num_iter,resize=2,images_min=heatmaps_raw_min,images_max=heatmaps_raw_max)
                            
                            # keypoints
                            self.summary.add_images('4.1 keypoints',images[:int(B/2)].cpu(), self.num_iter, mode='keypoints_conf', resize=2, keypoints=keypoints[:int(B/2)].cpu(), label=labels[:int(B/2)].cpu())
                            self.summary.add_images('4.2 keypoints_tps',images[int(B/2):].cpu(), self.num_iter, mode='keypoints_conf', resize=2, keypoints=keypoints[int(B/2):].cpu(), label=labels[int(B/2):].cpu())

                            # scoremap
                            self.summary.add_images('5.1 scoremap', network_diag['score_map'].cpu(), self.num_iter, resize=2)
                            self.summary.add_images('5.2 filtered scoremap', network_diag['filtered_score_map'].cpu(), self.num_iter, mode='keypoints_conf',resize=2, keypoints=keypoints.detach().clone())
                            
                            # tsne
                            self.summary.add_tsne('6. tsne', features_all_batch.cpu(), labels_all_batch.cpu(), prototype_norm, self.num_iter)

                            # alpha decomposition
                            if 'rgb' in network_diag.keys():
                                self.summary.add_images('7.1 rgb', network_diag['rgb'][0].cpu(), self.num_iter, resize=2)
                                self.summary.add_images('7.2 alpha', network_diag['alpha'][0].cpu(), self.num_iter, resize=2)

                        self.summary.flush()
                    # save model
                    if save_flag & self.training_config.checkpoint.save_weights:
                        self.save_model(network, network_solver, prototype_solver)
                if self.ddp:
                    dist.barrier()    
                # update counter
                self.num_iter += 1

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)



def train(rank, world_size, config): 
    # init ddp
    if world_size>1:
        setup(rank,world_size)
        print('DDP training: init cuda {} ...'.format(rank))
    else:
        print('Single GPU training')
    # init network

    network = NetWork(config.model)
    network = network.to(rank)
    
    # use ddp wrapper for distributed training
    if world_size>1:
        network = DataParallelPassthrough(network,device_ids=[rank],find_unused_parameters=True)

    # init solver
    network_solver = torch.optim.Adam(list(network.encoder.parameters())+list(network.decoder.parameters()), lr=config.training.lr.feature)
    prototype_solver = torch.optim.Adam([network.prototype], lr=config.training.lr.prototype, betas=(0.5, 0.999))
    
    # init data loader
    dataset_tr, sampler_tr = dataset(config, 'train', rank, world_size)

    # init network trainer
    trainer = Trainer(config, world_size, rank)
    
    # resume model
    if config.training.checkpoint.resume:
        trainer.resume_model_solver(network, network_solver, prototype_solver)

    # train model
    trainer.train(network, dataset_tr, sampler_tr, network_solver, prototype_solver)

def main():
    # get main config
    config = get_config()

    # num of cuda device
    n_gpus = torch.cuda.device_count()

    # hard code num of nodes (ONLY SUPPORT SINGLE NODE TRAINING!)
    n_nodes = 1
    
    # calculate world size
    world_size = n_gpus * n_nodes

    # update lr for multiple gpu
    config.training.lr.feature = config.training.lr.feature * world_size
    config.training.lr.prototype = config.training.lr.prototype * world_size

    # show config
    print_config(config)

    # create saved model folder
    if config.training.checkpoint.save_weights:
        if not os.path.isdir(config.metadata.model_dir):
            os.makedirs(config.metadata.model_dir)
            
    # crate log folder
    if not os.path.isdir(config.viz.log_dir + '/' + config.metadata.name):
        os.makedirs(config.viz.log_dir + '/' + config.metadata.name)

    # call train function
    if n_gpus>1:
        print('{} cuda devices available, using distributed data parallel to train model'.format(n_gpus))
        mp.spawn(train, args=(world_size,config), nprocs=world_size, join=True)
    else:
        print('single gpu training')
        train(0,world_size,config)


if __name__ == '__main__':
    main()
    
