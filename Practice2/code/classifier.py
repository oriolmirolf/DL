import os
from collections import Counter
import copy
from PIL import Image
import json

import torch
import numpy as np
import pandas as pd
from torchvision import models, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn

def convert_to_builtin_type(obj):
    if isinstance(obj, dict):
        return {k: convert_to_builtin_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin_type(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    else:
        return obj


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
                    self.labels.append(class_label)

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

def compute_per_class_metrics(preds, labels, probs, num_classes):
    metrics = []

    for cls in range(num_classes):
        cls_mask = labels == cls
        pred_mask = preds == cls

        cls_total = cls_mask.sum().item()
        cls_correct = (preds[cls_mask] == cls).sum().item()

        tp = (pred_mask & cls_mask).sum().item()
        fp = (pred_mask & (~cls_mask)).sum().item()
        fn = ((~pred_mask) & cls_mask).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        accuracy = cls_correct / max(cls_total, 1)

        cls_probs = probs[pred_mask]
        if cls_probs.numel() > 0:
            avg_softmax = cls_probs[:, cls].mean().item()
            entropy = -(cls_probs * (cls_probs + 1e-8).log()).sum(dim=1)
            avg_entropy = entropy.mean().item()
        else:
            avg_softmax, avg_entropy = None, None

        metrics.append({
            "class": int(cls),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "avg_softmax": avg_softmax,
            "avg_entropy": avg_entropy,
        })
    return metrics

def analyse_unseen_predictions(preds, labels, probs):
    metrics = []
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    unseen_classes = 58
    seen_classes = 43
    for cls in range(unseen_classes):
        cls_mask = labels == cls
        preds_subset = preds[cls_mask]
        probs_subset = probs[cls_mask]
        percentage_list = []
        probability_list = []
        for seen_cls in range(seen_classes):
            mask = preds_subset == seen_cls
            if not mask.any():
                percentage = 0.0
                mean_prob = 0.0
            else:
                percentage = 100.0 * mask.sum() / cls_mask.sum()
                mean_prob = probs_subset[mask].mean()
            percentage_list.append(percentage)
            probability_list.append(mean_prob)
        metrics.append({
            "class": int(cls),
            "percentage": percentage_list,
            "probability": probability_list,
        })
    return metrics





def evaluate_dataloader(model, dataloader, device, num_classes, unseen=False):
    model.eval()

    correct = 0
    total = 0

    all_labels = []
    all_preds = []
    max_probs = []

    total = len(dataloader.dataset)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)


            max_prob, _ = torch.max(F.softmax(outputs, dim=1), dim=1)

            _, predicted = torch.max(outputs, 1)

            all_labels.append(labels.cpu())
            all_preds.append(predicted.cpu())
            max_probs.append(max_prob.cpu())

            correct += (predicted == labels).sum().item()

    if total == 0:
        return {
            "loss": None,
            "accuracy": None,
            "per_class": [],
        }

    eval_accuracy = correct / total
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    max_probs = torch.cat(max_probs)
    if unseen:
        unseen_analysis = analyse_unseen_predictions(
            all_preds, all_labels, max_probs
        )
        return {"metrics": unseen_analysis}
    else:
        per_class = compute_per_class_metrics(all_preds, all_labels, max_probs, num_classes, unseen=unseen)

        return {
            "accuracy": eval_accuracy,
            "per_class": per_class,
        }


def train_model(model, criterion, optimizer, train_dataloader, val_dataloader, test_dataloader, epochs, device, output_dir="./output"):
    epoch_metrics = []
    best_val_accuracy = 0.0
    best_model_state = None
    num_classes = 43
    decay_patience = 8
    decay_counter = 0

    for epoch in range(epochs):

        if epoch == 0:

            for name, param in model.named_parameters():
                if name.startswith("fc"):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif epoch == 5:
            for param in model.parameters():
                param.requires_grad = True

            for param_group in optimizer.param_groups:
                param_group['lr'] /= 2

        # ------------------------- TRAIN ----------------------------------
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss /= len(train_dataloader.dataset)
        train_accuracy = correct / total

        # ------------------------- VALIDATE --------------------------------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_labels.append(labels.cpu())
                all_preds.append(predicted.cpu())
                all_probs.append(probs.cpu())

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_dataloader.dataset)
        val_accuracy = correct / total

        # Store metrics for this epoch
        epoch_metrics.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        # Track best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state = copy.deepcopy(model.state_dict())
            decay_counter = 0
        else:
            decay_counter += 1
            if decay_counter >= decay_patience:
                for param_group in optimizer.param_groups:
                    param_group['lr'] /= 2
                decay_counter = 0


        print(
            f"Epoch [{epoch}/{epochs}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
        )

    # ------------------------- TEST --------------------------------------
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_labels.append(labels.cpu())
            all_preds.append(predicted.cpu())
            all_probs.append(probs.cpu())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_dataloader.dataset)
    test_accuracy = correct / total

    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    all_probs = torch.cat(all_probs)
    test_per_class = compute_per_class_metrics(all_preds, all_labels, all_probs, num_classes)

    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

    # Save best model and metrics
    torch.save(best_model_state, output_dir+"/best_classifier.pth")
    with open(output_dir+"/epoch_metrics.json", "w") as f:
        json.dump(epoch_metrics, f, indent=2)
    with open(output_dir+"/test_metrics.json", "w") as f:
        json.dump({
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "per_class": test_per_class,
        }, f, indent=2)

    return model


def main():
    seed = 0
    lr = 1e-3
    output_dir = "output"
    epochs = 100
    batch_size = 128

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    df_train = pd.read_csv("data/Train.csv")
    df_train = df_train[["Path", "ClassId"]]

    df_test = pd.read_csv("data/Test.csv")
    df_test = df_test[["Path", "ClassId"]]

    
    df_train, df_val = train_test_split(
        df_train,
        test_size=0.2,
        stratify=df_train["ClassId"],
        random_state=seed
    )

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])


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
    test_dataloader = DataLoader(
        CsvImageDataset(df_test, transform=transform),
        batch_size=batch_size, 
        shuffle=False,          
        num_workers=4          
    )

    unseen_dataloader = DataLoader(
        OutDistributionDataset(transform=transform),
        batch_size=batch_size, 
        shuffle=False,          
        num_workers=4          
    )

    model = models.resnet50()
    model.load_state_dict(torch.load("./resnet50_weights.pth"))
    
    model.fc = torch.nn.Linear(model.fc.in_features, 43)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    os.makedirs(output_dir, exist_ok=True)

    model = train_model(
        model, criterion, optimizer,
        train_dataloader, val_dataloader, test_dataloader,
        epochs, device
    )

    model.load_state_dict(torch.load("./best_classifier.pth"))
    
    unseen_results = evaluate_dataloader(
        model, unseen_dataloader, device, 43, unseen=True
    )

    with open("./output/unseen_metrics.json", "w") as f:
        json.dump(convert_to_builtin_type(unseen_results), f, indent=4)
    

if __name__ == "__main__":
    main()