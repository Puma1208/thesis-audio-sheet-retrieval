# Implementatino for parallelization from:
# https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py


import os
import socket

from modules.MSMD_Dataset import MSMD_Dataset
from modules.CNN_GRU import CNN_GRU
from modules.Triplet_Contrastive_Loss import Triplet_Contrastive_Loss
from modules.MSMD_Trainer import MSMD_Trainer_Torchrun_Tracking
import torch
import time
import json
from torch.utils.data import DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch import nn
import torch.distributed as dist

def ddp_setup():
    
    init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    

def main(save_every:int, batch_size:int, total_epochs:int, snapshot_path:str='snapshot.pt'):
    ddp_setup()


    n_performances = 4
    snippets_batch_size = 15
    lr = .05
    K_negative=10
    margin=.05


    # Load train objects: dataset, model, optimizer
    msmd_images_dir = '/home/I6194030/MSMD_SNIPPETS_2'
    train_split_file = json.load(open('data/msmd_train_split.json'))
    validation_split_file = json.load(open('data/msmd_validation_split.json'))


    msmd_dataset_train = MSMD_Dataset(dir=msmd_images_dir, pieces=train_split_file, n_performances=n_performances, snippets_batch_size=snippets_batch_size)
    msmd_loader_train =  DataLoader(msmd_dataset_train, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(msmd_dataset_train))
    msmd_dataset_validation= MSMD_Dataset(dir=msmd_images_dir, pieces=validation_split_file, n_performances=n_performances, snippets_batch_size=snippets_batch_size)
    msmd_loader_validation =  DataLoader(msmd_dataset_validation, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(msmd_dataset_validation))
    model = CNN_GRU(1, 1, 32, 32, 128, 128, 64)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)#, weight_decay=.005)
    loss = Triplet_Contrastive_Loss(margin=margin, K_negative=K_negative)

    
    experiment_details = f'{batch_size}_{total_epochs}_{n_performances}_{snippets_batch_size}_{lr}_{K_negative}_{margin}'
    print(f'Experiment details: {experiment_details}')
    msmd_trainer = MSMD_Trainer_Torchrun_Tracking(model=model, 
                                train_data=msmd_loader_train,
                                validation_data=msmd_loader_validation,
                                optimizer=optimizer,
                                loss_function=loss,
                                save_every=save_every,
                                experiment_details=experiment_details)
    msmd_trainer.train(total_epochs)
    destroy_process_group()
    

if __name__ == '__main__':
    """
    main
    """

    timer_start = time.perf_counter()

    print(f'device count: {torch.cuda.device_count()}')
    save_every = 5
    batch_size = 64
    total_epochs = 50
    main(save_every=save_every, total_epochs=total_epochs, batch_size=batch_size)#, snapshot_path='snapshot_2.pt')

    timer_end = time.perf_counter()
    print(f'Entire time taken: {timer_end-timer_start:.4f} seconds')


