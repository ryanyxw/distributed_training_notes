"""
torchrun helps abstract a lot of boiler plate code and helps fault-tolerance by automatically continuign the training job form the previous checkpoint
https://pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html

Supplementary Material: https://pytorch.org/docs/stable/elastic/run.html
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

def ddp_setup():
    # got rid of ports, world sizes, ranks, etc. -> now it's just a simple setup telling it the backend
    init_process_group(backend="nccl")

class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            save_every: int,
            snapshot_path: str, #the path to the snapshot
    ) -> None:
        #torchrun provides us with a local environment variable that represents the rank of the current process
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        #we want to keep track of the number of epochs run in our snapshot
        self.epochs_run = 0
        if os.path.exists(snapshot_path):
            #if snapshot exists, load from snapshot_path
            print("loading snapshot")
            self._load_snapshot(snapshot_path)
        # wrap the model to DDP
        self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        """
        loads the model from the snapshot, along with which epoch it is on
        """
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

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
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        torch.save(snapshot, "snapshot.pt")
        print(f"Epoch {epoch} | Training checkpoint saved at snapshot.pt")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
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


def main(total_epochs, save_every, batch_size, snapshot_path: str = "snapshot.pt"):
    # create the process group and set master ip and port
    ddp_setup()

    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)
    trainer.train(total_epochs)

    #destroy the process group
    destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='torchrun distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    main(args.save_every, args.total_epochs)