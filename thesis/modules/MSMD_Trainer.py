# https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py#L64
# https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu_torchrun.py

import os
import torch
import json
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class MSMD_Trainer:
    def __init__(
            self,
            model:torch.nn.Module,
            train_data: DataLoader,
            optimizer:torch.optim.Optimizer,
            loss_function:torch.nn.Module,
            gpu_id:int,
            save_every:int
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data=train_data
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])


    def train(self, max_epochs:int):
        # print(f'Train mode: {self.model.module.training}')
        for epoch in range(max_epochs):
            self.run_epoch(epoch)
            if self.gpu_id==0 and epoch%self.save_every==0:
                self.save_checkpoint(epoch)

    # def eval(self, max_epochs:int):
    #     for epoch in range(max_epochs):
    #         self.model.module.eval()
    #         print(f'EVAL: Train mode: {self.model.module.training}')



    def run_epoch(self, epoch):
        batch_size = len(next(iter(self.train_data))[0])
        print(f'GPU {self.gpu_id} | EPOCH {epoch} | Batch size {batch_size} | Steps {len(self.train_data)}')
        self.train_data.sampler.set_epoch(epoch)
        for i, data in tqdm(enumerate(self.train_data), total=len(self.train_data)):
            snippets, excerpts = data
            snippets = snippets.to(self.gpu_id)#.to(torch.float32)
            excerpts = excerpts.to(self.gpu_id)
            # print(f'    GPU {self.gpu_id} | EPOCH {epoch} | Batch size {batch_size} | Steps {i}/{len(self.train_data)}')
            self.run_batch(snippets.to(torch.float32), excerpts.to(torch.float32))

    def run_batch(self, snipepts, excerpts):
        # print(f'RUN BATCH: {self.model.module.training}')
        # self.model.train()
        self.optimizer.zero_grad()
        output_network_score, output_network_audio = self.model(snipepts, excerpts)
        output_score = output_network_score[0]
        output_audio = output_network_audio[0]

        loss = self.loss_function(output_score, output_audio)
        loss.backward()
        self.optimizer.step()
        # self.eval(10)

    def save_checkpoint(self, epoch):
        model_checkpoint = self.model.module.state_dict()
        PATH = 'checkpoint.pt'
        torch.save(model_checkpoint, PATH)
        print(f'    EPOCH {epoch} | Training checkpoint saved at {PATH}')

class MSMD_Trainer_Torchrun:
    def __init__(
            self,
            model:torch.nn.Module,
            train_data: DataLoader,
            optimizer:torch.optim.Optimizer,
            loss_function:torch.nn.Module,
            save_every:int,
            snapshot_path,
            summary_writer_name
    ) -> None:
        self.gpu_id = int(os.environ['LOCAL_RANK'])
        self.model = model.to(self.gpu_id)
        self.train_data=train_data
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print(f'Loading Snapshot')
            self.load_snapshot(snapshot_path=snapshot_path)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        # self.writer = SummaryWriter(f'runs/msmd_experiments_1_{self.gpu_id}')
        self.writer = SummaryWriter(f'runs/summary_writer_{self.gpu_id}_{summary_writer_name}')

        self.writer.flush()


    def train(self, max_epochs:int):
        # print(f'Train mode: {self.model.module.training}')
        for epoch in range(self.epochs_run, max_epochs):
            self.run_epoch(epoch)
            if self.gpu_id==0 and epoch%self.save_every==0:
                self.save_snapshot(epoch)
        self.writer.close()

    # def eval(self, max_epochs:int):
    #     for epoch in range(max_epochs):
    #         self.model.module.eval()
    #         print(f'EVAL: Train mode: {self.model.module.training}')


    def run_epoch(self, epoch):
        batch_size = len(next(iter(self.train_data))[0])
        print(f'GPU {self.gpu_id} | EPOCH {epoch} | Batch size {batch_size} | Steps {len(self.train_data)}')
        self.train_data.sampler.set_epoch(epoch)
        losses_epoch = 0
        for i, data in enumerate(self.train_data):#, total=len(self.train_data):
            snippets, excerpts = data
            snippets = snippets.to(self.gpu_id)#.to(torch.float32)
            excerpts = excerpts.to(self.gpu_id)
            # print(f'    GPU {self.gpu_id} | EPOCH {epoch} | Batch size {batch_size} | Steps {i}/{len(self.train_data)}')
            losses_epoch += self.run_batch(snippets.to(torch.float32), excerpts.to(torch.float32))

        epoch_loss = losses_epoch/i+1
        print(f'Epoch loss: {epoch_loss}')
        self.writer.add_scalars('Training Loss', {'Triplet Loss': epoch_loss}, epoch)


    def run_batch(self, snipepts, excerpts):
        # print(f'RUN BATCH: {self.model.module.training}')
        self.optimizer.zero_grad()
        output_network_score, output_network_audio = self.model(snipepts, excerpts)
        output_score = output_network_score[0]
        output_audio = output_network_audio[0]

        loss = self.loss_function(output_score, output_audio)
        loss.backward()
        self.optimizer.step()
        return loss
        # self.eval(10)

    def save_snapshot(self, epoch):
        snapshot={
            'MODEL_STATE': self.model.module.state_dict(),
            'EPOCHS_RUN': epoch,
        }

        torch.save(snapshot, self.snapshot_path)
        print(f'Epoch {epoch} | Training snapshot saved at snapshot.pt')

    def load_snapshot(self, snapshot_path):
        loc = f'cuda:{self.gpu_id}'
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot['MODEL_STATE'])
        self.epochs_run = snapshot['EPOCHS_RUN']
        print(f'Resuming training from snapshot at Epoch {self.epochs_run}')

class MSMD_Trainer_Torchrun_Tracking:
    def __init__(
            self,
            model:torch.nn.Module,
            train_data: DataLoader,
            validation_data: DataLoader,
            optimizer:torch.optim.Optimizer,
            loss_function:torch.nn.Module,
            save_every:int,
            experiment_details:str
    ) -> None:
        self.gpu_id = int(os.environ['LOCAL_RANK'])
        self.model = model.to(self.gpu_id)
        self.train_data=train_data
        self.validation_data = validation_data
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = f'snapshot_GPU_{self.gpu_id}_{experiment_details}.pt'



        self.snapshot_path = self.snapshot_path
        if os.path.exists(self.snapshot_path):
            print(f'Loading Snapshot')
            self.load_snapshot(snapshot_path=self.snapshot_path)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.train_losses = []
        self.validation_losses = []
        os.makedirs('metrics_DDP', exist_ok=True)

        self.experiments_metrics = []
        self.metric_file = f'metrics_DDP/{self.gpu_id}_{experiment_details}.json'

        # self.writer = SummaryWriter(f'runs/2_summary_writer_name{self.gpu_id}')
        self.writer = SummaryWriter(f'runs/summary_writer_{self.gpu_id}_{experiment_details}')
        self.writer.flush()


    def train(self, max_epochs:int):
        # print(f'Train mode: {self.model.module.training}')
        for epoch in range(self.epochs_run, max_epochs):
            self.run_epoch(epoch)
            if self.gpu_id==0 and (epoch+1)%self.save_every==0:
                self.save_snapshot(epoch)
                # self.save_metrics()
        self.writer.close()
        self.snippets_excerpts_df = pd.DataFrame(self.experiments_metrics)
        self.snippets_excerpts_df.to_csv(self.metric_file)

    # def eval(self, max_epochs:int):
    #     for epoch in range(max_epochs):
    #         self.model.module.eval()
    #         print(f'EVAL: Train mode: {self.model.module.training}')


    def run_epoch(self, epoch):
        batch_size = len(next(iter(self.train_data))[0])
        print(f'GPU {self.gpu_id} | EPOCH {epoch} | Batch size {batch_size} | Steps {len(self.train_data)}')
        self.train_data.sampler.set_epoch(epoch)
        losses_epoch_train = 0
        losses_epoch_validation = 0

        # Training
        self.model.train()

        for i, data in enumerate(self.train_data):#, total=len(self.train_data):
            snippets, excerpts = data
            snippets = snippets.to(self.gpu_id)#.to(torch.float32)
            excerpts = excerpts.to(self.gpu_id)
            # print(f'    GPU {self.gpu_id} | EPOCH {epoch} | Batch size {batch_size} | Steps {i}/{len(self.train_data)}')
            l_train = self.run_batch_train(snippets.to(torch.float32), excerpts.to(torch.float32))
            losses_epoch_train += l_train
            if i%200==0:
                print(f' {self.gpu_id} TRAIN LOSS {i}/{len(self.train_data)} Last loss: {l_train}')

        epoch_loss_train = losses_epoch_train/len(self.train_data)

        # Validation
        self.model.eval()
        with torch.inference_mode():
            for i, data in enumerate(self.validation_data):
                snippets, excerpts = data
                snippets = snippets.to(self.gpu_id)
                excerpts = excerpts.to(self.gpu_id)

                l_validation = self.run_batch_eval(snippets.to(torch.float32), excerpts.to(torch.float32))
                losses_epoch_validation += l_validation
                if i%200==0:
                    print(f' {self.gpu_id} VALIDATION LOSS {i}/{len(self.validation_data)} Last loss: {l_validation}')
            epoch_loss_validation = losses_epoch_validation/len(self.validation_data)


        # print(f'    GPU {self.gpu_id} | EPOCH {epoch} | Epoch loss: {epoch_loss}')
        self.train_losses.append(epoch_loss_train)
        self.validation_losses.append(epoch_loss_validation)
        self.save_metrics()

        self.writer.add_scalars('Training Loss', {'Triplet Loss': epoch_loss_train, 'Validation TL': epoch_loss_validation}, epoch)
        self.epochs_run += 1


    def run_batch_train(self, snipepts, excerpts):
        # print(f'RUN BATCH: {self.model.module.training}')
        self.optimizer.zero_grad()
        output_network_score, output_network_audio = self.model(snipepts, excerpts)
        output_score = output_network_score[0]
        output_audio = output_network_audio[0]

        loss = self.loss_function(output_score, output_audio)
        loss.backward()
        self.optimizer.step()
        return loss.item()
        # self.eval(10)

    def run_batch_eval(self, snippets, excerpts):
        v_output_score, v_output_audio = self.model(snippets, excerpts)
        validation_loss = self.loss_function(v_output_score[0], v_output_audio[0])
        return validation_loss.item()

    def save_snapshot(self, epoch):
        snapshot={
            'MODEL_STATE': self.model.module.state_dict(),
            'EPOCHS_RUN': epoch,
        }

        torch.save(snapshot, self.snapshot_path)
        print(f'Epoch {epoch} | Training snapshot saved at {self.snapshot_path}')

    def load_snapshot(self, snapshot_path):
        loc = f'cuda:{self.gpu_id}'
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot['MODEL_STATE'])
        self.epochs_run = snapshot['EPOCHS_RUN']
        print(f'Resuming training from snapshot at Epoch {self.epochs_run}')

    def save_metrics(self):
        print(f'    GPU {self.gpu_id} FROM LOSSES: {self.train_losses} -> {self.train_losses[-1]}')
        self.experiments_metrics.append({'gpu':self.gpu_id,
                                        'epochs_run':self.epochs_run,
                                        'training_loss':self.train_losses[-1],
                                        'validation_loss':self.validation_losses[-1]
                                        })
