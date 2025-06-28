import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision import transforms as T
from src.datasets.patch_dataset import PatchDataset
from src.datasets.simclr_dataset import SimCLRDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# Reference: https://arxiv.org/abs/2002.05709
# SimCLR: A Simple Framework for Contrastive Learning of Visual Representations

class SimCLRModel(nn.Module):
    def __init__(self, base_model="resnet18", out_dim=128):
        super().__init__()
        self.encoder = getattr(models, base_model)(pretrained=False)
        dim_mlp = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim)
        )

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projector(features)
        return projections # we're only using the feature extractor part

def nt_xent_loss(z_i, z_j, temperature=0.5):
    """ 
    Computes the contrastive loss for two sets of embeddings.
    Parameters:
        z_i (torch.Tensor): Embeddings from the first view, shape (B, D).
        z_j (torch.Tensor): Embeddings from the second view, shape (B, D).
        temperature (float): Temperature parameter for scaling the similarity.
    """
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature
    N = z.shape[0]
    labels = torch.arange(N).to(z.device)
    labels = (labels + N // 2) % N
    mask = ~torch.eye(N, dtype=bool).to(z.device)
    sim = sim.masked_select(mask).view(N, -1)
    return F.cross_entropy(sim, labels)

def get_simclr_transform():
    return T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

def pretrain_simclr(patch_dir, epochs=10, batch_size=128, lr=1e-3):
    base_transform = get_simclr_transform()
    base_dataset = PatchDataset(patch_dir, transform=None)
    simclr_dataset = SimCLRDataset(base_dataset, transform=base_transform)
    dataloader = DataLoader(simclr_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimCLRModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for x_i, x_j in tqdm(dataloader, desc=f"SimCLR Epoch {epoch+1}"):
            x_i, x_j = x_i.to(device), x_j.to(device)
            z_i = model(x_i)
            z_j = model(x_j)
            loss = nt_xent_loss(z_i, z_j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "simclr_encoder.pth")
    print("[INFO] SimCLR pretraining complete.")

