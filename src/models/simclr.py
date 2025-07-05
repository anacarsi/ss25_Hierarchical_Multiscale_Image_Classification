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
        self.encoder = getattr(models, base_model)(weights=None)
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
    NT-Xent loss implementation without invalid labels for cross_entropy.
    """
    device = z_i.device
    N = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)  # (2N, D)
    z = F.normalize(z, dim=1)

    sim_matrix = torch.matmul(z, z.T)  # (2N, 2N)
    sim_matrix = sim_matrix / temperature

    # Mask out self-similarity
    mask = torch.eye(2 * N, dtype=torch.bool).to(device)
    sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

    # Positive similarities (i paired with i+N and vice versa)
    positives = torch.cat([torch.diag(sim_matrix, N), torch.diag(sim_matrix, -N)]).unsqueeze(1)  # (2N, 1)

    # Denominator: logsumexp over all other similarities
    denominator = torch.logsumexp(sim_matrix, dim=1, keepdim=True)  # (2N, 1)

    loss = -positives + denominator
    return loss.mean()


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

def pretrain_simclr(patch_dir, epochs=200, batch_size=512, lr=1e-3):
    base_transform = get_simclr_transform()
    base_dataset = PatchDataset(patch_dir, transform=None)
    simclr_dataset = SimCLRDataset(base_dataset, transform=base_transform)
    dataloader = DataLoader(simclr_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SimCLRModel().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop_patience = 20
    best_epoch = -1
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
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # Early stopping 
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            best_epoch = epoch + 1
            # Optionally save best model so far
            torch.save(model.state_dict(), "simclr_encoder_best.pth")
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 20 == 0:
            print(f"[INFO] Early stopping check: {epochs_no_improve} epochs without improvement (patience={early_stop_patience})")
            if epochs_no_improve >= early_stop_patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch+1}. Best epoch: {best_epoch} with loss {best_loss:.4f}")
                break

        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint_path = f"simclr_encoder_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[INFO] SimCLR checkpoint saved: {checkpoint_path}")

    torch.save(model.state_dict(), "simclr_encoder.pth")
    print("[INFO] SimCLR pretraining complete.")

