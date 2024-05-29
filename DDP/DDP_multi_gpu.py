"""
Simplest boilerplate for DDP with multi-gpu training
https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset

#This is a wrapper around python's native multiprocessing library
import torch.multiprocessing as mp

#This helps distribute data to each gpu
from torch.utils.data.distributed import DistributedSampler

#This is the actual DDP library
from torch.nn.parallel import DistributedDataParallel as DDP

#initializes and destroy our distributed process group -> allows the gpus to discover each other
from torch.distributed import init_process_group, destroy_process_group

import os

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes

    """
    #master port is the one that coordinates across all gpus
    os.environ["MASTER_ADDR"] = "localhost" # this is the IP address of the machine running the rank 0
    os.environ["MASTER_PORT"] = "12355" # this is the port that rank 0 is listening to
    torch.cuda.set_device(rank) # set the current device to the rank
    #initialize default process group with nccl -> nvidia collective communication library -> used for distributed communications
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            gpu_id: int,
            save_every: int,
    ) -> None:
        self.gpu_id = self.gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        # wrap the model to DDP
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])

        #this ensures that the data is shuffled properly across epochs when distributed
        self.train_data.sampler.set_epoch(epoch)

        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        # For DDP, self.model is now a DDP instance, and you access the original model parameters thru "module" to save it
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            # models across gpus should be synchronized after each "step", so we don't need to wait for any processes
            # thus we can directly check if we're currently the main process and save the checkpoint (doens't have to be main)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)


def load_train_objs():
    train_set = MyTrainDataset(2048)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        #the distributed sampler knows how to read the current rank and world size to assign data properly?
        sampler=DistributedSampler(dataset), #add the distributed sampler to dataloader so it knows how to distribute data
    )


def main(rank, world_size, total_epochs, save_every, batch_size):
    # create the process group and set master ip and port
    ddp_setup(rank, world_size)

    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, save_every)
    trainer.train(total_epochs)

    #destroy the process group
    destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    #counts the number of cudas available
    world_size = torch.cuda.device_count()

    #mp.spawm will automatically give a rank for each process
    mp.spawn(main, args=(world_size, args.total_epochs, args.save_every, args.batch_size), nprocs=world_size)
    # main(device, args.total_epochs, args.save_every, args.batch_size)