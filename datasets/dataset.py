import torch
import torch.utils.data

from datasets.mnist import MNIST
from datasets.celeba import CelebA
from datasets.h36m import H36M
from datasets.multi_object import MultiObject

def create_dataloader(dataset,batch_size,rank, world_size):
    if world_size > 1:
        print('creating ddp data loader ...')
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank)
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=batch_size,
                                                num_workers=4,
                                                pin_memory=True,
                                                sampler=sampler)
    else:
        print('creating single process data loader ...')
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=4,
                                                pin_memory=True)
        sampler = None
    return loader, sampler

def dataset(config, mode, rank=0, world_size=1, evaluate=False):
    if config.data.dataset.name.startswith('mnist'):
        dataset = MNIST(config.data, mode, config.training.loss.feature.eqvar.activate)
        loader, sampler = create_dataloader(dataset, config.training.batch_size, rank, world_size)
    elif config.data.dataset.name.startswith('Celeb') or config.data.dataset.name.startswith('MAFL'):
        dataset = CelebA(config.data, mode, config.training.loss.feature.eqvar.activate)
        loader, sampler = create_dataloader(dataset, config.training.batch_size, rank, world_size)
    elif config.data.dataset.name.startswith('H36M'):  
        dataset = H36M(config.data, mode, config.training.loss.feature.eqvar.activate)
        loader, sampler = create_dataloader(dataset, config.training.batch_size, rank, world_size)
    elif config.data.dataset.name.startswith('multi_object'):  
        dataset = MultiObject(config.data, mode, config.training.loss.feature.eqvar.activate, evaluate)
        loader, sampler = create_dataloader(dataset, config.training.batch_size, rank, world_size) 
    else:
        raise Exception('invalid dataset')
    
    return loader, sampler
