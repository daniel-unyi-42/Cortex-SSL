# for few-shot learning, we used pretrained=True, frozen=False and num_labeled=1/7/14/21/28/35
# and compared the results with a randomly initialized model: pretrained=False, frozen=False and num_labeled=1/7/14/21/28/35

# for linear probing, we used pretrained=True, frozen=True and num_labeled=70
# and compared the results with a randomly initialized model: pretrained=False, frozen=False and num_labeled=70

# in all cases, we used 10 samples for validation and 21 samples for testing

####################################################################################################

import argparse
import pandas as pd
import numpy as np
import pyvista
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
import torch_geometric.transforms as T
import torch_geometric.nn as gnn
from sklearn.metrics import confusion_matrix

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Segmentation model training script")
    parser.add_argument('--pretrained', type=bool, default=True, help='Whether to use a pretrained model')
    parser.add_argument('--frozen', type=bool, default=True, help='Whether to freeze the pretrained model')
    parser.add_argument('--num_labeled', type=int, default=7, help='Number of labeled samples to use')
    args = parser.parse_args()
    return args

args = parse_args()

pretrained = args.pretrained
frozen = args.frozen
num_labeled = args.num_labeled

# Other hyperparameters
batch_size = 8
epochs = 500
hidden = 64
lr = 0.005 if frozen else 0.001

log_path = f'runs/seg/linearprobing/pretrain_{pretrained}' if frozen else f'runs/seg/{num_labeled}-shot/pretrain_{pretrained}'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seeds for reproducibility
torch.manual_seed(123)
np.random.seed(123)

# Define custom loss and metrics functions
def get_dice_loss(y_pred, y_true):
    num_classes = y_pred.shape[1]
    y_true = F.one_hot(y_true, num_classes=num_classes)
    y_pred = F.softmax(y_pred, dim=1)
    loss = 0.0
    for i in range(num_classes):
        loss += -2.0 * torch.sum(y_true[:,i] * y_pred[:,i]) / torch.sum(y_true[:,i] + y_pred[:,i])
    loss /= num_classes
    return loss

def get_iou_per_class(cm):
    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    return TP / (TP + FP + FN)

def get_iou(cm):
    return np.mean(get_iou_per_class(cm))

def get_dice_per_class(cm):
    iou = get_iou_per_class(cm)
    return 2 * iou / (iou + 1)

def get_dice(cm):
    iou = get_iou(cm)
    return 2 * iou / (iou + 1)

def get_acc(cm):
    TP = np.diag(cm)
    return np.sum(TP) / np.sum(cm)

# load dataset
dataset = []
transform = T.Compose([T.GenerateMeshNormals(), T.FaceToEdge(), T.NormalizeScale()])

# Label mapping dictionary
y_dict = {
    0:0, 1:0, 2:1, 3:2, 4:0, 5:3, 6:4, 7:5, 8:6, 9:7, 10:8, 11:9, 12:10, 13:11,
    14:12, 15:13, 16:14, 17:15, 18:16, 19:17, 20:18, 21:19, 22:20, 23:21, 24:22,
    25:23, 26:24, 27:25, 28:26, 29:27, 30:28, 31:29, 32:0, 33:0, 34:30, 35:31
}

# Load data from files
feature_dir = 'Mindboggle101_mindboggle_output_tables_20141017/'
surface_dir = 'surface_labels/'

df = pd.read_csv('subjects.txt', sep=', ')

for num, subject in enumerate(df['Mindboggle101'], start=1):
    reader = pyvista.get_reader(surface_dir + subject + '/lh.labels.DKT31.manual.vtk')
    mesh = reader.read()
    assert mesh.is_manifold
    # assert mesh.is_all_triangles # not true for all meshes, why?
    pos = mesh.points
    face = mesh.faces.reshape(-1, 4)[:, 1:]  # Extract the faces from the mesh
    x_df = pd.read_csv(f'{feature_dir}{subject}/left_surface/vertices.csv')
    x = x_df[["FreeSurfer thickness", "FreeSurfer curvature", "FreeSurfer convexity (sulc)"]].to_numpy()
    y = np.array([y_dict[val] for val in mesh.get_array('Labels')])

    # Ensure data consistency
    assert pos.shape[0] == face.max() + 1
    assert pos.shape[0] == x.shape[0]
    assert pos.shape[0] == y.shape[0]

    data = Data(
        x=torch.from_numpy(x).to(torch.float32),
        face=torch.from_numpy(np.ascontiguousarray(face.T)).to(torch.int64),
        pos=torch.from_numpy(pos).to(torch.float32),
        y=torch.from_numpy(y).to(torch.int64),
        name=subject
    )
    data = transform(data)
    dataset.append(data)
    print(num, subject, data)

# Shuffle dataset
np.random.shuffle(dataset)

# Create data loaders
train_loader = DataLoader(dataset[:num_labeled], batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset[70:80], batch_size=batch_size)
test_loader = DataLoader(dataset[80:], batch_size=batch_size)

# Define neural network components
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
        self.mod4 = GNModule(hidden, hidden // 2)
        self.x_mlp = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 9),
        )
        self.adj_mlp = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        self.optimizer = None

    def forward(self, data):
        x = torch.cat([data.x, data.pos, data.norm], dim=1)
        x = self.mod1(x, data.edge_index)
        x = self.mod2(x, data.edge_index)
        x = self.mod3(x, data.edge_index)
        x = self.mod4(x, data.edge_index)
        return x

class GCN(nn.Module):
    def __init__(self, pretrained, frozen, hidden):
        super().__init__()
        self.backbone = AutoEncoder(hidden)
        if pretrained:
            self.backbone.load_state_dict(torch.load('pretrained_finalstandard.pt'))
            if frozen:
                for param in self.backbone.parameters():
                    param.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(hidden // 2, hidden // 2),
            nn.ReLU(),
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
    def test(self, device, loader):
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

model = Segmentor(pretrained, frozen, hidden).to(device)
model.optimizer = torch.optim.Adam(model.parameters(), lr=lr) if frozen else torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
print(model)

train_losses, train_accs, train_ious = [], [], []
val_losses, val_accs, val_ious = [], [], []
best_val_index = 0

for epoch in range(epochs):
    loss, cm = model.train_step(device, train_loader)
    train_losses.append(loss)
    train_accs.append(get_acc(cm))
    train_ious.append(get_dice(cm))
    loss, cm = model.test(device, val_loader)
    val_losses.append(loss)
    val_accs.append(get_acc(cm))
    val_ious.append(get_dice(cm))
    if loss < val_losses[best_val_index]:
        best_val_index = epoch
        torch.save(model.state_dict(), f'{log_path}/model.pt')
    print(
        epoch, train_losses[-1], train_accs[-1], train_ious[-1],
        val_losses[-1], val_accs[-1], val_ious[-1]
    )

# Load the best model and evaluate on the test set
model.load_state_dict(torch.load(f'{log_path}/model.pt'))
loss, cm = model.test(device, test_loader)
acc = get_acc(cm)
iou = get_dice(cm)
print(loss, acc, iou)
