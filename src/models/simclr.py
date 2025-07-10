import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision import transforms as T
from src.datasets.patch_dataset import PatchDataset
from src.datasets.simclr_dataset import SimCLRDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime

# Reference: https://arxiv.org/abs/2002.05709
# SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
# 10.07 Trying to solve overfitting: added Gaussian blur, random rotation, and vertical flip to the augmentations


class SimCLRModel(nn.Module):
    def __init__(self, base_model="resnet18", out_dim=128):
        super().__init__()
        self.encoder = getattr(models, base_model)(weights=None)
        dim_mlp = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, out_dim)
        )

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projector(features)
        return projections  # we're only using the feature extractor part


def nt_xent_loss(z_i, z_j, temperature=0.5):
    device = z_i.device
    N = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)  # (2N, D)
    z = F.normalize(z, dim=1)

    sim_matrix = torch.matmul(z, z.T) / temperature  # (2N, 2N)

    mask = torch.eye(2 * N, dtype=torch.bool).to(device)
    # Use a large negative value compatible with the current dtype
    if sim_matrix.dtype == torch.float16:
        neg_value = -65504.0  # min value for float16
    else:
        neg_value = -1e9
    sim_matrix = sim_matrix.masked_fill(mask, neg_value)

    positives = torch.cat(
        [torch.diag(sim_matrix, N), torch.diag(sim_matrix, -N)]
    ).unsqueeze(
        1
    )  # (2N, 1)

    denominator = torch.logsumexp(sim_matrix, dim=1, keepdim=True)  # (2N, 1)

    if torch.isnan(denominator).any():
        print("[ERROR] NaN in denominator!")
    if torch.isnan(positives).any():
        print("[ERROR] NaN in positives!")

    loss = -positives + denominator
    mean_loss = loss.mean()

    if torch.isnan(mean_loss):
        print("[ERROR] Final loss is NaN!")

    return mean_loss


def get_simclr_transform():
    return T.Compose(
        [
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(), 
            T.RandomRotation(90),   
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)), # gaussian blur for simclr
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

def pretrain_simclr(patch_dir, epochs=200, batch_size=512, lr=1e-3, level=3):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    base_transform = get_simclr_transform()
    base_dataset = PatchDataset(patch_dir, transform=None) 
    simclr_dataset = SimCLRDataset(base_dataset, transform=base_transform)

    dataloader = DataLoader(
        simclr_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )

    print(f"SimCLR dataset length: {len(simclr_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimCLRModel().to(device)

    # Add Weight Decay to optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5) 
    # Add Learning Rate Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=15, verbose=True, min_lr=1e-6) 

    best_loss = float("inf")
    epochs_no_improve = 0
    early_stop_patience = 30 # patience for pretraining
    best_epoch = -1
    scaler = torch.cuda.amp.GradScaler() # this allows for mixed precision training
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for step, (x_i, x_j) in enumerate(
            tqdm(dataloader, desc=f"SimCLR Epoch {epoch+1}")
        ):
            x_i, x_j = x_i.to(device), x_j.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(): 
                z_i = model(x_i)
                z_j = model(x_j)
                loss = nt_xent_loss(z_i, z_j)

            if torch.isnan(loss): # skip this step if loss is NaN
                print(f"[ERROR] Loss is NaN at step {step}!")
                continue

            try:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            except RuntimeError as e:
                print(f"[ERROR] Backward pass failed at step {step}: {e}")
                continue
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, SimCLR Loss: {avg_loss:.4f}")

        scheduler.step(avg_loss) # step the scheduler based on average loss

        # Early stopping
        now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            best_epoch = epoch + 1
            # Save best model so far
            best_model_path = f"src/models/simclr_encoder_best_level{level}_{now}.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"[INFO] Best SimCLR model saved with loss: {best_loss:.4f} at {best_model_path}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            print(
                f"[INFO] Early stopping triggered at epoch {epoch+1}. Best epoch: {best_epoch} with loss {best_loss:.4f}"
            )
            break

        # Save checkpoint every 50 epochs (or adjust frequency)
        if (epoch + 1) % 50 == 0:
            now_ckpt = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            checkpoint_path = f"src/models/simclr_encoder_epoch{epoch+1}_level{level}_{now_ckpt}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[INFO] SimCLR checkpoint saved: {checkpoint_path}")

    now_final = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    final_model_path = f"src/models/simclr_encoder_level{level}_{now_final}.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"[INFO] SimCLR pretraining complete. Final model saved: {final_model_path}.")
