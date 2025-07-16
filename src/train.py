import torch
from tqdm import tqdm


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    DEBUG = "\033[96m"
    INFO = "\033[95m"  # pink
    WARNING = "\033[93m"  # yellow
    ERROR = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def train_resnet(
    model,
    train_loader,
    val_loader,
    train_dataset,
    val_dataset,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    base_model_path,
):
    """
    Generic training loop for a ResNet model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_val_acc = 0.0
    epochs_no_improve = 0
    early_stop_patience = 10
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct = 0, 0
        for imgs, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
        train_acc = correct / len(train_dataset)

        # Validation
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
        val_acc = val_correct / len(val_loader.dataset)

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}"
        )
        scheduler.step(val_acc)

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"{base_model_path}_best.pth")
            print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Best model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(
                    f"{bcolors.INFO}[INFO]{bcolors.ENDC} Early stopping triggered at epoch {epoch+1}."
                )
                break

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{base_model_path}_epoch{epoch+1}.pth")

    torch.save(model.state_dict(), f"{base_model_path}_final.pth")
    print(f"{bcolors.INFO}[INFO]{bcolors.ENDC} Final model saved.")
