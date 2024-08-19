import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from NN import MAE

def load_train_objs(device, lr, weight_decay, hidden):
    transform = T.Compose([T.GenerateMeshNormals(), T.FaceToEdge(), T.NormalizeScale()])
    # load HCP dataset
    HCP_dataset = []
    for subject in os.listdir('HCP_data'):
        surface = nib.load(f'HCP_data/{subject}/T1w/Native/{subject}.L.pial.native.surf.gii')
        pos, face = surface.agg_data()
        pos = torch.from_numpy(pos).to(torch.float32)
        face = torch.from_numpy(face.T).contiguous().to(torch.long)
        thickness = nib.load(f'HCP_data/{subject}/MNINonLinear/Native/{subject}.L.thickness.native.shape.gii').agg_data()
        curvature = nib.load(f'HCP_data/{subject}/MNINonLinear/Native/{subject}.L.curvature.native.shape.gii').agg_data()
        sulc = nib.load(f'HCP_data/{subject}/MNINonLinear/Native/{subject}.L.sulc.native.shape.gii').agg_data()
        x = torch.from_numpy(np.stack([thickness, curvature, sulc], axis=1)).to(torch.float32)
        data = Data(pos=pos, face=face, x=x)
        data = transform(data)
        data.subject = subject
        HCP_dataset.append(data)
    np.random.shuffle(HCP_dataset)
    HCP_train_set = HCP_dataset[:1000]
    HCP_val_set = HCP_dataset[1000:]
    # initialize model
    model = MAE(hidden).to(device)
    model.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model = DDP(model, device_ids=[device])
    return HCP_train_set, HCP_val_set, model

def main(batch_size, total_epochs, lr, weight_decay, hidden):
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    init_process_group(backend="nccl", init_method='env://', rank=rank, world_size=world_size)
    device = int(os.environ['SLURM_LOCALID'])
    torch.cuda.set_device(device)
    train_set, val_set, model = load_train_objs(device, lr, weight_decay, hidden)
    train_sampler = DistributedSampler(
        train_set,
        num_replicas=world_size,
        rank=rank,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=train_sampler
    )
    val_sampler = DistributedSampler(
        val_set,
        num_replicas=world_size,
        rank=rank,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=val_sampler
    )
    for epoch in range(total_epochs):
        train_x_loss, train_adj_loss = model.module.train_step(device, train_loader)
        val_x_loss, val_adj_loss = model.module.val_step(device, val_loader)
        print(f'Node: {rank} | GPU: {device} | Epoch: {epoch} | Train X Loss: {train_x_loss:.4f} | Train Adj Loss: {train_adj_loss:.4f} | Val X Loss: {val_x_loss:.4f} | Val Adj Loss: {val_adj_loss:.4f}')
        if rank == 0:
            torch.save(model.module.state_dict(), 'pretrained_finalstandard.pt')
    destroy_process_group()

if __name__ == "__main__":
    batch_size = 8
    lr = 0.001
    weight_decay = 0.01
    total_epochs = 500
    hidden = 64
    main(batch_size, total_epochs, lr, weight_decay, hidden)
