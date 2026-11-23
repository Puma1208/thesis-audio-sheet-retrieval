"""
Train the CNN-GRU audio-sheet music model with a triplet contrastive loss,
using PyTorch DistributedDataParallel

Usage from the repository root:
torchrun --nproc_per_node=<n_gpus> -m thesis.scripts.train --data_dir <snippets_dir>
    
Setup for parallelization based on:
https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py
"""

import argparse
import os

from thesis.modules.MSMD_Dataset import MSMD_Dataset
from thesis.modules.CNN_GRU import CNN_GRU
from thesis.modules.Triplet_Contrastive_Loss import Triplet_Contrastive_Loss
from thesis.modules.MSMD_Trainer import MSMD_Trainer_Torchrun_Tracking
import torch
import time
import json
from torch.utils.data import DataLoader

from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

def ddp_setup():
    if torch.cuda.is_available():
        init_process_group(backend='nccl')
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        # CPU fallback for quick test run
        init_process_group(backend='gloo')

def main(data_dir:str, train_split:str, validation_split:str, save_every:int, batch_size:int, total_epochs:int):
    ddp_setup()


    n_performances = 4
    snippets_batch_size = 15
    lr = .05
    K_negative=10
    margin=.05


    # Load train objects: dataset, model, optimizer
    msmd_images_dir = data_dir
    with open(train_split) as f:
        train_split_file = json.load(f)
    with open(validation_split) as f:
        validation_split_file = json.load(f)


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

    parser = argparse.ArgumentParser(description='Train the CNN-GRU audio-sheet music model (launch with torchrun).')
    parser.add_argument('--data_dir', required=True, help='directory with the snippets created by create_dataset.py')
    parser.add_argument('--train_split', default='data/splits/msmd_train_split.json')
    parser.add_argument('--validation_split', default='data/splits/msmd_validation_split.json')
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--total_epochs', type=int, default=50)
    args = parser.parse_args()

    timer_start = time.perf_counter()

    print(f'device count: {torch.cuda.device_count()}')
    main(data_dir=args.data_dir,
         train_split=args.train_split,
         validation_split=args.validation_split,
         save_every=args.save_every,
         total_epochs=args.total_epochs,
         batch_size=args.batch_size)

    timer_end = time.perf_counter()
    print(f'Entire time taken: {timer_end-timer_start:.4f} seconds')
