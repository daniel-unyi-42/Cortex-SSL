import os
import copy
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import torch_geometric.nn as gnn
from torch_geometric.utils import negative_sampling, k_hop_subgraph

def augment_graph(data, num_nodes_to_delete, num_hops=10):
    data = copy.deepcopy(data)
    nodes_to_delete = torch.randint(0, data.pos.shape[0], (num_nodes_to_delete,))
    subset, _, _, _ = k_hop_subgraph(nodes_to_delete, num_hops, data.edge_index)
    mask = torch.ones((data.pos.shape[0], 1), device=data.edge_index.device).detach()
    mask[subset] = 0.0
    data.x = (data.x * mask).float()
    data.pos = (data.pos * mask).float()
    data.norm = (data.norm * mask).float()
    mask = mask.squeeze().bool()
    edge_mask = (mask[data.edge_index[0]]) & (mask[data.edge_index[1]])
    data.edge_index = data.edge_index[:, edge_mask]
    return data

class GNModule(nn.Module):
    def __init__(self, indim, outdim):
        super(GNModule, self).__init__()
        self.conv = gnn.ChebConv(indim, outdim, K=3)
        self.act = nn.PReLU()
        self.bn = gnn.BatchNorm(outdim)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.act(x)
        x = self.bn(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, hidden):
        super(AutoEncoder, self).__init__()
        self.mod1 = GNModule(9, hidden)
        self.mod2 = GNModule(hidden, hidden)
        self.mod3 = GNModule(hidden, hidden)
        self.mod4 = GNModule(hidden, hidden//2)
        self.x_mlp = nn.Sequential(
            nn.Linear(hidden//2, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 9),
        )
        self.adj_mlp = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1),
        )
        self.optimizer = None

    def forward(self, data):
        x = torch.cat([data.x, data.pos, data.norm], dim=1)
        x = self.mod1(x, data.edge_index)
        x = self.mod2(x, data.edge_index)
        x = self.mod3(x, data.edge_index)
        x = self.mod4(x, data.edge_index)
        return x
    
    def loss(self, data):
        x = torch.cat([data.x, data.pos, data.norm], dim=1)
        aug_data = augment_graph(data, 100, 10)
        z = self(aug_data)
        return self.x_loss(x, z), self.adj_loss(data.edge_index, z)
    
    def x_loss(self, x, z):
        return F.mse_loss(x, self.x_mlp(z))
    
    def adj_loss(self, edge_index, z):
        EPS = 1e-15
        pos_rec = torch.sigmoid(self.adj_mlp(torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)))
        pos_loss = -torch.log(pos_rec + EPS).mean()
        neg_edge_index = negative_sampling(edge_index, z.size(0))
        neg_rec = torch.sigmoid(self.adj_mlp(torch.cat([z[neg_edge_index[0]], z[neg_edge_index[1]]], dim=1)))
        neg_loss = -torch.log(1 - neg_rec + EPS).mean()
        return pos_loss + neg_loss

    def train_step(self, device, loader):
        self.train()
        x_losses = []
        adj_losses = []
        for data in loader:
            self.optimizer.zero_grad()
            data = data.to(device)
            x_loss, adj_loss = self.loss(data)
            (x_loss + adj_loss).backward()
            self.optimizer.step()
            x_losses.append(x_loss.item())
            adj_losses.append(adj_loss.item())
        return sum(x_losses) / len(x_losses), sum(adj_losses) / len(adj_losses)
    
    @torch.no_grad()
    def val_step(self, device, loader):
        self.eval()
        x_losses = []
        adj_losses = []
        for data in loader:
            data = data.to(device)
            x_loss, adj_loss = self.loss(data)
            x_losses.append(x_loss.item())
            adj_losses.append(adj_loss.item())
        return sum(x_losses) / len(x_losses), sum(adj_losses) / len(adj_losses)
    
    @torch.no_grad()
    def predict_step(self, device, test_set):
        self.eval()
        zs = []
        ys = []
        for data in test_set:
            data = data.to(device)
            z = self(data)
            zs.append(z)
            ys.append(data.y)
        return torch.cat(zs, dim=0).cpu().numpy(), torch.cat(ys, dim=0).cpu().numpy()

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
    model = AutoEncoder(hidden).to(device)
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
