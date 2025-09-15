from __future__ import annotations

import json
import argparse
import os
from pathlib import Path
from typing import List
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import ConcatDataset
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import open_clip

CLASS_NAMES: dict[int, str] = {
    0: "speed limit 5 km/h",
    1: "speed limit 15 km/h",
    2: "speed limit 30 km/h",
    3: "speed limit 40 km/h",
    4: "speed limit 50 km/h",
    5: "speed limit 60 km/h",
    6: "speed limit 70 km/h",
    7: "speed limit 80 km/h",
    8: "no straight or left turn",
    9: "no straight or right turn",
    10: "no straight ahead",
    11: "no left turn",
    12: "no left or right turn",
    13: "no right turn",
    14: "no overtaking from left",
    15: "no U‑turn",
    16: "no cars allowed",
    17: "no horn",
    18: "speed limit 40 km/h",
    19: "speed limit 50 km/h",
    20: "go straight or right",
    21: "go straight ahead",
    22: "go left",
    23: "go left or right",
    24: "go right",
    25: "keep left",
    26: "keep right",
    27: "roundabout ahead",
    28: "watch out for cars",
    29: "horn allowed",
    30: "bicycle crossing",
    31: "U‑turn allowed",
    32: "road divider ahead",
    33: "traffic signals ahead",
    34: "danger ahead",
    35: "zebra crossing",
    36: "bicycle crossing",
    37: "children crossing",
    38: "dangerous curve to the left",
    39: "dangerous curve to the right",
    40: "steep downhill slope ahead warning",
    41: "uphill slope ahead warning sign",
    42: "slow down",
    43: "go right or straight",
    44: "go left or straight",
    45: "residential area",
    46: "zigzag curve",
    47: "train crossing",
    48: "under construction",
    49: "winding road ahead",
    50: "fences ahead",
    51: "heavy‑vehicle accident warning",
    52: "stop sign",
    53: "give way",
    54: "no stopping",
    55: "no entry",
    56: "yield to oncoming traffic ahead",
    57: "acces control",
}


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

        return image, (1,label)
    
    def load_image(self, img_path):
        return Image.open(img_path).convert('RGB')


class OutDistributionDataset(Dataset):

    def __init__(
        self,
        root_path: str | Path,
        transform=None,
        exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    ) -> None:
        self.root_path = Path(root_path)
        self.transform = transform
        self.exts = tuple(e.lower() for e in exts)

        self.image_paths: List[Path] = []
        self.labels: List[int] = []

        # Scan 0‑57 sub‑folders
        for class_id in range(58):
            class_dir = self.root_path / str(class_id)
            if not class_dir.is_dir():
                continue

            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.exts and img_path.is_file():
                    self.image_paths.append(img_path)
                    self.labels.append(class_id)

        if not self.image_paths:
            raise RuntimeError(f"No images found under {self.root_path}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, (0, self.labels[idx]),

class OutDistributionDatasetTest(Dataset):

    def __init__(
        self,
        root_path: str | Path,
        transform=None,
        exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp"),
    ) -> None:
        self.root_path = Path(root_path)
        self.transform = transform
        self.exts = tuple(e.lower() for e in exts)

        self.image_paths: List[Path] = []
        self.labels: List[int] = []

        class_dir = self.root_path

        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in self.exts and img_path.is_file():
                class_id_str = img_path.stem[:3]
                class_id = int(class_id_str)
                self.image_paths.append(img_path)
                self.labels.append(class_id)

        if not self.image_paths:
            raise RuntimeError(f"No images found under {self.root_path}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, (0, self.labels[idx]),

class ConvAutoencoder(nn.Module):

    def __init__(self, latent_dim: int = 128):
        super().__init__()

        # ------- Encoder -------
        self.enc_conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 224 → 112
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 112 → 56
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 56 → 28
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),# 28 → 14
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, 4, stride=2, padding=1),# 14 → 7
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
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


class OOD_System:
    def __init__(self, args):

        self.autoencoder = ConvAutoencoder()
        self.autoencoder.load_state_dict(torch.load("./best_autoencoder.pth"))
        self.autoencoder.to("cuda")
        self.autoencoder.eval()

        self.classifier = models.resnet50()
        
        self.classifier.fc = torch.nn.Linear(self.classifier.fc.in_features, 43)
        self.classifier.to("cuda")

        self.classifier.load_state_dict(torch.load("./best_classifier.pth"))
        self.classifier.eval()

        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            args.model, pretrained=args.ckpt, device='cuda'
        )

        self.to_pil = transforms.ToPILImage()

        self.clip_model.eval()

        self.mse = nn.MSELoss(reduction="none")

        tokenizer = open_clip.get_tokenizer(args.model)

        self.text_features = self.encode_text_prompts(self.clip_model,
                                                      tokenizer,
                                                      device='cuda')

    @staticmethod
    @torch.no_grad()
    def encode_text_prompts(model, tokenizer, device) -> torch.Tensor:
        prompts = [f"a photo of a {desc} traffic sign" for desc in CLASS_NAMES.values()]
        text_tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    @staticmethod
    @torch.no_grad()
    def clip_predict(model, img_tensor: torch.Tensor,
                     text_features: torch.Tensor) -> torch.Tensor:

        image_features = model.encode_image(img_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        logits = image_features @ text_features.T
        return logits.softmax(dim=-1)


    @torch.no_grad()
    def __call__(self, img_tensor: torch.Tensor) -> int:
        reconstructed_image = self.autoencoder(img_tensor)
        err = self.mse(img_tensor, reconstructed_image).view(img_tensor.size(0), -1).mean(dim=1)

        if err < 0.00828: # Precomputed threshold
            logits = self.classifier(img_tensor)
            _, predicted = torch.max(logits, 1)
            return (1, int(predicted.item()))
        else:
            pil = self.to_pil(img_tensor.squeeze(0).cpu())
            clip_tensor = self.preprocess(pil).to(img_tensor.device).unsqueeze(0)
            probs = self.clip_predict(self.clip_model, clip_tensor, self.text_features)
            return (0, int(probs.argmax(dim=-1).item()))
    
    def evaluate_in_distribution(self, img_tensor):
        logits = self.classifier(img_tensor)
        _, predicted = torch.max(logits, 1)
        return (1, predicted)

    def evaluate_out_distribution(self, img_tensor):
        probs = self.clip_predict(self.clip_model, img_tensor, self.text_features)
        return (0, int(probs.argmax(dim=-1).item()))

@torch.no_grad()
def evaluate_in_distribution_only(
    OOD: OOD_System,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
):

    total, correct = 0, 0
    y_true, y_pred = [], []

    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        _, cls_ids = labels
        cls_ids = cls_ids.to(device)

        for img, cls in zip(imgs, cls_ids):
            pred_flag, pred_cls = OOD.evaluate_in_distribution(img.unsqueeze(0))
            assert pred_flag == 1
            total   += 1
            correct += int(pred_cls == cls.item())
            y_true.append(cls.item())
            y_pred.append(pred_cls.cpu())

    acc  = correct / total
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"[IID-only]   accuracy : {correct}/{total} = {acc:.2%}")
    print(f"[IID-only]   precision: {prec:.2%}   recall: {rec:.2%} f1: {f1:.2%}")
    
    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}

    with open("metrics_in_distribution_experiment.json", "w") as fp:
        json.dump(metrics, fp, indent=4)


@torch.no_grad()
def evaluate_out_distribution_only(
    OOD: OOD_System,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda",
):

    total_det, correct_det = 0, 0
    total_cls, correct_cls = 0, 0
    y_true, y_pred = [], []

    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        _, cls_ids = labels
        cls_ids = cls_ids.to(device)

        for img, cls in zip(imgs, cls_ids):
            pred_flag, pred_cls = OOD.evaluate_out_distribution(img.unsqueeze(0))

            total_det   += 1
            correct_det += int(pred_flag == 0)

            total_cls   += 1
            correct_cls += int(pred_cls == cls.item())
            y_true.append(cls.item())
            y_pred.append(pred_cls)

    det_acc = correct_det / total_det
    cls_acc = correct_cls / total_cls
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    print(f"[OOD-only] CLIP acc / prec / rec / f1: {cls_acc:.2%}  {prec:.2%}  {rec:.2%} {f1:.2%}")
    metrics = {
        "detection_accuracy": det_acc,
        "clip_accuracy": cls_acc,
        "precision": prec,
        "recall": rec,
    }
    with open("metrics_out_distribution_experiment.json", "w") as fp:
        json.dump(metrics, fp, indent=4)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OOD traffic‑sign recognition with CLIP fallback")
    parser.add_argument(
        "--data",
        type=str,
        default="./data_out_distribution/traffic_Data/DATA",
        help="Root directory containing 0…57 sub‑folders.",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="./data_out_distribution/traffic_Data/TEST",
        help="Root directory containing Test images.",
    )
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--model", type=str, default="ViT-B-32", help="CLIP backbone (open_clip)")
    parser.add_argument(
        "--ckpt", type=str, default="laion2b_s34b_b79k", help="Pretrained checkpoint name"
    )
    return parser.parse_args()


from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json
import pandas as pd

def evaluate_full_system(OOD, dataloader, device, num_classes=101):
    total_cls, correct_cls = 0, 0
    y_true_cls, y_pred_cls = [], []

    total_det, correct_det = 0, 0
    y_true_det, y_pred_det = [], []

    total_sys, correct_sys = 0, 0
    # NEW: collect for full-system confusion / F1
    y_true_sys, y_pred_sys = [], []

    rows_for_csv = []

    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        dist_flags, class_ids = labels
        dist_flags, class_ids = dist_flags.to(device), class_ids.to(device)

        for img, dist_flag, cls_id in zip(imgs, dist_flags, class_ids):
            img = img.unsqueeze(0)
            is_iid_pred, pred_label = OOD(img)

            # ─── OOD detection metrics ─────────────────────────────────────────
            total_det   += 1
            correct_det += int(is_iid_pred == dist_flag.item())
            y_true_det.append(dist_flag.item())
            y_pred_det.append(is_iid_pred)

            # ─── IID classification metrics ────────────────────────────────────
            if is_iid_pred and dist_flag.item():  # only for in-domain accepted
                total_cls   += 1
                correct_cls += int(pred_label == cls_id.item())
                y_true_cls.append(cls_id.item())
                y_pred_cls.append(pred_label)

            # ─── Full-system accuracy ──────────────────────────────────────────
            total_sys += 1
            detect_ok = (is_iid_pred == dist_flag.item())
            class_ok  = (not is_iid_pred) or (pred_label == cls_id.item())
            correct_sys += int(detect_ok and class_ok)

            # NEW: record every sample's true vs. predicted label
            y_true_sys.append(cls_id.item())
            y_pred_sys.append(pred_label)

            rows_for_csv.append({
                "file_idx"          : len(rows_for_csv),
                "ground_label"      : cls_id.item(),
                "pred_label"        : pred_label,
                "distribution_label": dist_flag.item(),
                "is_iid_pred"       : is_iid_pred
            })

    metrics = {}

    # IID classification
    if total_cls > 0:
        accuracy  = correct_cls / total_cls
        precision = precision_score(y_true_cls, y_pred_cls,
                                    average='macro', zero_division=0)
        recall    = recall_score(y_true_cls, y_pred_cls,
                                    average='macro', zero_division=0)
        f1        = f1_score(y_true_cls, y_pred_cls,
                             average='macro', zero_division=0)

        metrics["iid_classification"] = {
            "total"           : total_cls,
            "correct"         : correct_cls,
            "accuracy"        : accuracy,
            "precision_macro" : precision,
            "recall_macro"    : recall,
            "f1_macro"        : f1
        }

        print(f"IID classification accuracy : {correct_cls}/{total_cls} = {accuracy:.2%}")
        print(f"Precision (macro)           : {precision:.2%}")
        print(f"Recall   (macro)            : {recall:.2%}")
        print(f"F1       (macro)            : {f1:.2%}")
    else:
        print("No sample qualified for IID classification (all were OOD or mis-flagged).")

    # OOD detection
    precision_det = precision_score(y_true_det, y_pred_det, zero_division=0)
    recall_det    = recall_score(y_true_det, y_pred_det, zero_division=0)
    f1_det        = f1_score(y_true_det, y_pred_det, zero_division=0)
    det_acc       = correct_det / total_det

    metrics["ood_detection"] = {
        "total"          : total_det,
        "correct"        : correct_det,
        "accuracy"       : det_acc,
        "precision_macro": precision_det,
        "recall_macro"   : recall_det,
        "f1_macro"       : f1_det
    }

    print(f"OOD‐detection accuracy : {correct_det}/{total_det} = {det_acc:.2%}")
    print(f"OOD‐detection F1       : {f1_det:.2%}")

    # Full‐system accuracy
    sys_acc = correct_sys / total_sys
    metrics["full_system"] = {
        "total"    : total_sys,
        "correct"  : correct_sys,
        "accuracy" : sys_acc
    }
    print(f"Full‐system accuracy   : {correct_sys}/{total_sys} = {sys_acc:.2%}")

    # ─── NEW: Overall macro-F₁ ─────────────────────────────────────────────
    # build confusion matrix over all 101 classes
    cm_sys = confusion_matrix(
        y_true_sys,
        y_pred_sys,
        labels=list(range(num_classes))
    )
    # compute per-class F1 and the macro-average
    f1_per_class    = f1_score(y_true_sys, y_pred_sys,
                               labels=list(range(num_classes)),
                               average=None,
                               zero_division=0)
    f1_macro_system = f1_score(y_true_sys, y_pred_sys,
                               average='macro',
                               zero_division=0)

    metrics["full_system"].update({
        "f1_per_class" : f1_per_class.tolist(),
        "f1_macro"     : f1_macro_system
    })

    print(f"Full‐system macro-F1    : {f1_macro_system:.2%}")

    cm_det = confusion_matrix(y_true_det, y_pred_det)
    disp_det = ConfusionMatrixDisplay(confusion_matrix=cm_det,
                                      display_labels=["OOD", "IID"])
    disp_det.plot(cmap=plt.cm.Blues)
    plt.title("OOD Detection Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix_ood_detection_in_system.png")

    # Save JSON + per-image CSV
    with open("metrics_complete_experiment.json", "w") as fp:
        json.dump(metrics, fp, indent=4)
    pd.DataFrame(rows_for_csv).to_csv("per_image_results.csv", index=False)

    print("Saved metrics_complete_experiment.json and per_image_results.csv")

    return metrics




def main() -> None:
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    OOD = OOD_System(args)

    df_in_distribution = pd.read_csv("data/Test.csv")
    df_in_distribution = df_in_distribution[["Path", "ClassId"]]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset_in_distribution = CsvImageDataset(df_in_distribution, transform=transform)

    dataset_out_distribution_train = OutDistributionDataset(root_path=args.data, transform=transform)
    dataset_out_distribution_test = OutDistributionDatasetTest(root_path=args.test_data, transform=transform)

    combined_dataset = ConcatDataset([dataset_in_distribution, dataset_out_distribution_test])

    in_distribution_loader = DataLoader(
        dataset_in_distribution,
        batch_size=args.batch,
        shuffle=False,
        num_workers=os.cpu_count() // 2 or 1,
        pin_memory=(device == "cuda"),
    )

    out_distribution_loader = DataLoader(
        dataset_out_distribution_train,
        batch_size=args.batch,
        shuffle=False,
        num_workers=os.cpu_count() // 2 or 1,
        pin_memory=(device == "cuda"),
    )

    combined_loader = DataLoader(
        combined_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=os.cpu_count() // 2 or 1,
        pin_memory=(device == "cuda"),
    )

    #evaluate_in_distribution_only(OOD, in_distribution_loader, device)
    evaluate_out_distribution_only(OOD, out_distribution_loader, device)
    evaluate_full_system(OOD, combined_loader, device)


if __name__ == "__main__":
    main()