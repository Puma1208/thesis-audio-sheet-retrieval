# https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu.py#L64
# https://github.com/pytorch/examples/blob/main/distributed/ddp-tutorial-series/multigpu_torchrun.py

import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

class MSMD_Trainer_Torchrun_Tracking:
    """
    Trains the CNN-GRU with DPP, runs the validation every epoch, 
    saves 
    - the snapshots and logs in train/validation
    - losses to metrics_DPP/
    - and TensorBoard in runs/
    """
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
        self.device = torch.device(f'cuda:{self.gpu_id}') if torch.cuda.is_available() else torch.device('cpu')
        self.model = model.to(self.device)
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
        self.model = DDP(self.model, device_ids=[self.gpu_id] if torch.cuda.is_available() else None)
        self.train_losses = []
        self.validation_losses = []
        os.makedirs('metrics_DDP', exist_ok=True)

        self.experiments_metrics = []
        self.metric_file = f'metrics_DDP/{self.gpu_id}_{experiment_details}.csv'

        self.writer = SummaryWriter(f'runs/summary_writer_{self.gpu_id}_{experiment_details}')
        self.writer.flush()


    def train(self, max_epochs:int):
        # print(f'Train mode: {self.model.module.training}')
        for epoch in range(self.epochs_run, max_epochs):
            self.run_epoch(epoch)
            if self.gpu_id==0 and (epoch+1)%self.save_every==0:
                self.save_snapshot(epoch)
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
            snippets = snippets.to(self.device)#.to(torch.float32)
            excerpts = excerpts.to(self.device)
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
                snippets = snippets.to(self.device)
                excerpts = excerpts.to(self.device)

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
        snapshot = torch.load(snapshot_path, map_location=self.device)
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
