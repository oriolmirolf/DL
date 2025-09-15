import os
from collections import Counter
import copy
from PIL import Image
import json
import random

import torch
import numpy as np
import pandas as pd
from torchvision import models, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset, random_split



class CsvImageDataset(torch.utils.data.Dataset):

    def __init__(self, pandas_dataframe, transform=None):
        self.data_frame = pandas_dataframe
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = self.data_frame.iloc[idx, 0]
        img_path = "./data/" + img_path
        label = self.data_frame.iloc[idx, 1]
        image = self.load_image(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def load_image(self, img_path):
        return Image.open(img_path).convert('RGB')

class OutDistributionDataset(torch.utils.data.Dataset): 
    def __init__(self, transform=None):
        self.root_path = "./data_out_distribution/traffic_Data/DATA/"
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for class_label in range(58):
            class_folder = os.path.join(self.root_path, str(class_label))
            if not os.path.isdir(class_folder):
                continue 
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                if os.path.isfile(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(0)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = self.load_image(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label
    
    def load_image(self, img_path):
        return Image.open(img_path).convert('RGB')


#<--------- AUTOENCODER --------->

class ConvAutoencoder(nn.Module):

    def __init__(self, latent_dim: int = 128):
        super().__init__()

        # ------- Encoder -------
        self.enc_conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 224 → 112
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 112 → 56
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 56 → 28
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),# 28 → 14
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, 4, stride=2, padding=1),# 14 → 7
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.enc_fc = nn.Linear(512 * 7 * 7, latent_dim)

        # ------- Decoder -------
        self.dec_fc = nn.Linear(latent_dim, 512 * 7 * 7)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), # 7 → 14
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 14 → 28
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 28 → 56
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 56 → 112
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # 112 → 224
            nn.Sigmoid()  # out ∈ [0,1]
        )

    def encode(self, x):
        h = self.enc_conv(x)
        h = h.view(h.size(0), -1)
        z = self.enc_fc(h)
        return z

    def decode(self, z):
        h = self.dec_fc(z)
        h = h.view(h.size(0), 512, 7, 7)
        x_hat = self.dec_conv(h)
        return x_hat

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
    

def train_autoencoder(model, train_loader, val_loader, output_name,
                      epochs: int = 50,
                      lr: float = 1e-3,
                      device: torch.device = "cuda",
                      decay_patience: int = 8):

    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    patience = 0

    metrics = {
        "initial_val_loss": None,
        "epochs": []
    }

    # Initial validation loss
    model.eval()
    running_loss = 0.
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            x_hat = model(x)
            loss = criterion(x_hat, x)
            running_loss += loss.item() * x.size(0)
    val_loss = running_loss / len(val_loader.dataset)

    print(f"Initial val {val_loss:.4f}")
    metrics["initial_val_loss"] = val_loss

    for epoch in range(epochs):
        # -------- train ----------
        model.train()
        running_loss = 0.
        for x, _ in train_loader:
            x = x.to(device)
            opt.zero_grad()
            x_hat = model(x)
            loss = criterion(x_hat, x)
            loss.backward()
            opt.step()
            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # -------- valid ----------
        model.eval()
        running_loss = 0.
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device)
                x_hat = model(x)
                loss = criterion(x_hat, x)
                running_loss += loss.item() * x.size(0)
        val_loss = running_loss / len(val_loader.dataset)

        print(f"[{epoch:03d}] train {train_loss:.4f} | val {val_loss:.4f}")
        metrics["epochs"].append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        # early-stopping
        if val_loss < best_val - 1e-4:
            torch.save(model.state_dict(), output_name+ "_best_autoencoder.pth")
            best_val = val_loss
            patience = 0
        else:
            patience += 1
            if patience >= decay_patience:
                for param_group in opt.param_groups:
                    param_group['lr'] /= 2
                patience = 0

    # Save metrics to JSON
    with open(output_name + "_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

@torch.no_grad()
def evaluate_ood_hybrid(model, loader, mu, inv_cov,
                        alpha, beta, threshold, device="cuda"):

    mse = nn.MSELoss(reduction="none")
    model.eval().to(device)

    errs, dists, labels = [], [], []
    for x, y in loader:
        x = x.to(device)
        z  = model.encode(x)
        xh = model.decode(z)
        re = mse(xh, x).view(x.size(0), -1).mean(1)
        md = mahalanobis(z.cpu(), mu, inv_cov)
        errs.append(re.cpu());  dists.append(md);  labels.append(y.cpu())

    errs  = torch.cat(errs);   dists = torch.cat(dists);   labels = torch.cat(labels)

    novelty = alpha * dists + beta * errs            
    pred_id = (novelty <= threshold).int()
    acc  = accuracy_score(labels, pred_id)
    prec = precision_score(labels, pred_id, zero_division=0)
    rec  = recall_score(labels, pred_id, zero_division=0)
    f1   = f1_score(labels, pred_id, zero_division=0)
    return pred_id, labels, acc, prec, rec, f1


@torch.no_grad()
def collect_latent_vectors(model, loader, device="cuda"):
    zs = []
    model.eval()
    for x, _ in loader:
        z = model.encode(x.to(device)).cpu()
        zs.append(z)
    return torch.cat(zs, dim=0)          # (N , latent_dim)

def fit_latent_gaussian(z_train: torch.Tensor, eps: float = 1e-5):
    mu = z_train.mean(0)
    cov = torch.from_numpy(
        np.cov(z_train.numpy(), rowvar=False) + eps * np.eye(z_train.size(1))
    ).float()
    inv_cov = torch.inverse(cov)
    return mu, inv_cov

def mahalanobis(z: torch.Tensor, mu: torch.Tensor, inv_cov: torch.Tensor):
    diff = z - mu
    return torch.sqrt(torch.einsum('bi,ij,bj->b', diff, inv_cov, diff))

@torch.no_grad()
def show_random_reconstruction(model, dataloader, device="cuda"):
    model.eval()
    model.to(device)

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels
        break

    idx = random.randint(0, images.size(0) - 1)
    print(labels[idx])
    original = images[idx].unsqueeze(0)
    reconstruction = model(original)

    original = original.squeeze().cpu().permute(1, 2, 0).numpy()
    reconstruction = reconstruction.squeeze().cpu().permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(reconstruction)
    axes[1].set_title("Reconstruction")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

@torch.no_grad()
def compute_normalized_errors(autoencoder, dataloader, device, mu, inv_cov):
    mse = nn.MSELoss(reduction="none")
    errs, dists = [], []
    autoencoder.eval()
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device, non_blocking=True)
            z  = autoencoder.encode(x)
            xh = autoencoder.decode(z)
            errs.append(mse(xh, x).view(x.size(0), -1).mean(1).cpu())
            dists.append(mahalanobis(z.cpu(), mu, inv_cov))
    
    errs = torch.cat(errs)
    dists = torch.cat(dists)

    alpha = 1.0 / dists.std()
    beta  = 1.0 / errs.std()

    hybrid_score = alpha * dists + beta * errs

    return hybrid_score, alpha, beta

@torch.no_grad()
def compute_hybrid_scores(
        model: nn.Module,
        loader: torch.utils.data.DataLoader,
        mu: torch.Tensor,
        inv_cov: torch.Tensor,
        alpha: float,
        beta: float,
        device: str = "cuda"
    ) -> torch.Tensor:

    model = model.to(device).eval()
    mse = nn.MSELoss(reduction="none")

    all_scores = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        z  = model.encode(x)
        xh = model.decode(z)

        rec_err = mse(xh, x).view(x.size(0), -1).mean(1)

        md = mahalanobis(z.cpu(), mu, inv_cov)

        novelty = alpha * md + beta * rec_err.cpu()
        all_scores.append(novelty)

    return torch.cat(all_scores, dim=0)

def plot_error_hist(id_err,
                    ood_err=None,
                    *,
                    bins=100,
                    title="Error distribution on reconstruction",
                    save_path=None,
                    log_scale=False,
                    density=True):

    id_err = torch.as_tensor(id_err).detach().cpu().numpy()
    if ood_err is not None:
        ood_err = torch.as_tensor(ood_err).detach().cpu().numpy()

    plt.figure(figsize=(6, 4))
    plt.hist(id_err,
             bins=bins,
             alpha=.6,
             label="ID",
             density=density)
    if ood_err is not None:
        plt.hist(ood_err,
                 bins=bins,
                 alpha=.6,
                 label="OOD",
                 density=density)

    if log_scale:
        plt.yscale("log")

    plt.xlabel("Reconstruction error")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.grid(ls="--", alpha=.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# NO NEED TO RETRAIN THE AUTOENCODERS CAN BE USED THE VANILLA TRAINED ONES
def main():
    seeds = [0]
    lr = 1e-3
    output_dir = "./output_ae_improved/"
    os.makedirs(output_dir, exist_ok=True)
    epochs = 100
    batch_size = 128
    latent_dims = [4, 8, 16, 32, 64, 128]

    metrics = []
    for seed in seeds:
        for latent_dim in latent_dims:

            torch.manual_seed(seed)
            np.random.seed(seed)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            df_train = pd.read_csv("data/Train.csv")
            df_train = df_train[["Path", "ClassId"]]

            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

            
            df_train["ClassId"] = 1
            
            df_train, df_val = train_test_split(
                df_train,
                test_size=0.2,
                stratify=df_train["ClassId"],
                random_state=seed
            )

            df_train = df_train.reset_index(drop=True)
            df_val = df_val.reset_index(drop=True)

            unseen_dataset = OutDistributionDataset(transform=transform)

            train_unseen_size = int(0.8 * len(unseen_dataset))
            val_unseen_size = len(unseen_dataset) - train_unseen_size

            generator = torch.Generator().manual_seed(0)

            _, val_unseen_dataset = random_split(unseen_dataset, [train_unseen_size, val_unseen_size], generator=generator)

            train_dataloader = DataLoader(
                CsvImageDataset(df_train, transform=transform),
                batch_size=batch_size, 
                shuffle=True,          
                num_workers=4          
            )

            val_dataloader = DataLoader(
                CsvImageDataset(df_val, transform=transform),
                batch_size=batch_size,
                shuffle=False,
                num_workers=4
            )

            unseen_dataloader = DataLoader(
                val_unseen_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4
            )

            val_complete_dataset = ConcatDataset([CsvImageDataset(df_val, transform=transform), val_unseen_dataset])
            val_complete_dataloader = DataLoader(val_complete_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

            autoencoder = ConvAutoencoder(latent_dim=latent_dim).to(device)
            #autoencoder.load_state_dict(torch.load(output_dir+"best_autoencoder_reduced.pth"))
            #show_random_reconstruction(autoencoder, test_dataloader, device=device)

            output_name = output_dir + f'autoencoder_latent_dim_{latent_dim}_seed_{seed}'

            train_autoencoder(autoencoder, train_dataloader, val_dataloader, output_name,
                                epochs=epochs, lr=lr, device=device, decay_patience=5)
            # show_random_reconstruction(autoencoder, test_dataloader, device=device)
            
            autoencoder.load_state_dict(torch.load(output_name+ "_best_autoencoder.pth"))

            z_train = collect_latent_vectors(autoencoder, train_dataloader, device)
            mu, inv_cov = fit_latent_gaussian(z_train)

            id_scores, alpha, beta = compute_normalized_errors(
                autoencoder, val_dataloader, device, mu, inv_cov
            )

            ood_scores = compute_hybrid_scores(
                autoencoder, unseen_dataloader, mu, inv_cov, alpha, beta, device
            )

            plot_error_hist(id_scores,
                            ood_scores,
                            bins=80,
                            save_path=output_dir + f"val_latent_{latent_dim}_seed_{seed}.png",
                            log_scale=False)

            best_threshold = torch.quantile(id_scores, 0.95).item()

            id_pred, true_labels, accuracy, precision, recall, f1 = evaluate_ood_hybrid(
                autoencoder, val_complete_dataloader, mu, inv_cov,
                alpha, beta, best_threshold, device=device
            )

            print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            cm = confusion_matrix(true_labels, id_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["OOD (0)", "ID (1)"])
            disp.plot(cmap="Blues")
            plt.title("Confusion Matrix")
            plt.savefig(output_dir + f'autoencoder_latent_dim_{latent_dim}_seed_{seed}_confusion_matrix_accuracy_threshold.png')

            metrics.append({
                "seed": seed,
                "latent_dim": latent_dim,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "best_threshold": best_threshold
            })

            with open(os.path.join(output_dir, f'summary_info_latent_dim_{latent_dim}_seed_{seed}'), "w") as f:
                json.dump(metrics, f, indent=4)


# NO NEED TO RETRAIN THE AUTOENCODERS CAN BE USED THE VANILLA TRAINED ONES
if __name__ == "__main__":  
    main()