import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.utils import negative_sampling, k_hop_subgraph
from sklearn.metrics import confusion_matrix


def get_dice_loss(y_pred, y_true):
    num_classes = y_pred.shape[1]
    y_true = F.one_hot(y_true, num_classes=num_classes)
    y_pred = F.softmax(y_pred, dim=1)
    loss = 0.0
    for i in range(num_classes):
        loss += -2.0 * torch.sum(y_true[:,i] * y_pred[:,i]) / torch.sum(y_true[:,i] + y_pred[:,i])
    loss /= num_classes
    return loss


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


class MAE(nn.Module):
    def __init__(self, hidden):
        super(MAE, self).__init__()
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


class Segmentor(nn.Module):
    def __init__(self, pretrained, frozen, hidden):
        super().__init__()
        self.backbone = MAE(hidden)
        if pretrained:
            self.backbone.load_state_dict(torch.load('pretrained_finalstandard.pt'))
            if frozen: # linear probing
                for param in self.backbone.parameters():
                    param.requires_grad = False
                self.head = nn.Sequential(
                    nn.Linear(hidden // 2, 32),
                )
            else: # few-shot learning
                self.head = nn.Sequential(
                    nn.Linear(hidden // 2, hidden // 2),
                    nn.ReLU(),
                    nn.Linear(hidden // 2, 32),
                )
        self.optimizer = None

    def forward(self, data):
        x = self.backbone(data)
        return self.head(x)
    
    def train_step(self, device, loader):
        self.train()
        losses = []
        cms = []
        for data in loader:
            self.optimizer.zero_grad()
            data = data.to(device)
            out = self(data)
            loss = F.cross_entropy(out, data.y) + get_dice_loss(out, data.y)
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()
            pred = out.argmax(dim=1)
            cm = confusion_matrix(data.y.detach().cpu().numpy(), pred.detach().cpu().numpy())
            assert cm.shape[0] == 32 and cm.shape[1] == 32
            cms.append(cm)
        return sum(losses) / len(losses), sum(cms)
    
    @torch.no_grad()
    def test_step(self, device, loader):
        self.eval()
        losses = []
        cms = []
        for data in loader:
            data = data.to(device)
            out = self(data)
            loss = F.cross_entropy(out, data.y) + get_dice_loss(out, data.y)
            losses.append(loss.item())
            pred = out.argmax(dim=1)
            cm = confusion_matrix(data.y.detach().cpu().numpy(), pred.detach().cpu().numpy())
            assert cm.shape[0] == 32 and cm.shape[1] == 32
            cms.append(cm)
        return sum(losses) / len(losses), sum(cms)
