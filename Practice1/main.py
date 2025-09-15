import os
import time
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder

from models import MaorNet, ResNet18

class MAMeDataset(Dataset):

    def __init__(self, csv_file, img_dir, transform=None, subset=None):
        self.data = pd.read_csv(csv_file)
        label_encoder = LabelEncoder()
        self.data['target'] = label_encoder.fit_transform(self.data['Medium'])
        if subset is not None and 'Subset' in self.data.columns:
            self.data = self.data[self.data['Subset'] == subset]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        filename = row['Image file']
        label = row['target']
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def compute_normalization_values():

    csv_file = os.path.join("dataset", "MAMe_dataset.csv")
    img_dir = os.path.join("dataset", "imgs")

    temp_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    temp_train_dataset = MAMeDataset(csv_file=csv_file, img_dir=img_dir,
                                     transform=temp_transform, subset='train')
    temp_train_loader = DataLoader(temp_train_dataset, batch_size=64, shuffle=False)

    # Compute normalization values.
    mean, std = compute_mean_std(temp_train_loader)
    print("Computed normalization values:")
    print("Mean:", mean)
    print("Std:", std)

def compute_mean_std(loader):
    mean = 0.0
    std = 0.0
    total_images = 0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)  # flatten H x W
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples
    mean /= total_images
    std /= total_images
    return mean, std

def train_sgd(model, 
              train_loader, 
              val_loader, 
              test_loader, 
              device, 
              num_epochs=100, 
              lr=0.001, 
              momentum=0.0, 
              weight_decay=0.0, 
              lr_scheduling=False, 
              early_stopping=False):
    loss_list = []
    train_accuracy_list = []
    val_accuracy_list = []
    test_accuracy_list = []

    def eval_loader(loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                labels = labels.to(device, dtype=torch.long)
                preds = model(imgs).argmax(1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return 100. * correct / total
    
    val_acc  = eval_loader(val_loader)
    test_acc = eval_loader(test_loader)
    print(f"Validation Accuracy after epoch 0: {val_acc:.2f}%")
    print(f"Test Accuracy after epoch 0: {test_acc:.2f}%\n")
    val_accuracy_list.append(val_acc); test_accuracy_list.append(test_acc)
    train_accuracy.append(float(100/29))      # random baseline

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    best_val_accuracy = val_acc
    early_stop_counter = 0
    lr_schedule_counter = 0

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        loss_list.append(avg_loss)
        train_accuracy = 100 * correct / total
        train_accuracy_list.append(train_accuracy)

        val_acc    = eval_loader(val_loader)
        test_acc   = eval_loader(test_loader)

        loss_list.append(avg_loss)
        val_accuracy_list.append(val_acc)
        test_accuracy_list.append(test_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Loss: {avg_loss:.4f}  "
              f"Train: {train_accuracy:.2f}%  "
              f"Val: {val_acc:.2f}%  "
              f"Test: {test_acc:.2f}%")

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            early_stop_counter = 0
            lr_schedule_counter = 0
        else:
            early_stop_counter += 1
            lr_schedule_counter += 1

        if lr_scheduling and lr_schedule_counter >= 5:
            lr = lr * 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print(f"Learning rate decreased to {lr:.6f} after {lr_schedule_counter} epochs with no improvement.")
            lr_schedule_counter = 0

        if early_stopping and early_stop_counter >= 10:
            print("Early stopping triggered: no improvement in validation accuracy for 10 consecutive epochs.")
            break

    total_training_time = time.time() - start_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    metrics = {
        "loss": loss_list,
        "train_accuracy": train_accuracy_list,
        "val_accuracy": val_accuracy_list,
        "test_accuracy": test_accuracy_list,
        "training_time": total_training_time
    }
    return metrics

def train_sgd_no_overfit(
        model,
        train_loader, val_loader, test_loader, device,
        num_epochs        = 200,
        lr                = 0.05,      # should be rescaled by batch_size/256
        momentum          = 0.9,
        weight_decay      = 1e-4,
        early_stopping    = True,
        patience          = 10):

    
    loss_list, train_acc_list, val_acc_list, test_acc_list = [], [], [], []

    
    optimizer    = torch.optim.SGD(model.parameters(),
                                   lr=lr,
                                   momentum=momentum,
                                   weight_decay=weight_decay)

    
    warm_epochs  = 5
    warmup  = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-8, end_factor=1.0,
                total_iters=warm_epochs)

    cosine  = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_epochs - warm_epochs)

    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    
    best_val_acc, no_improve = 0.0, 0

    
    def eval_loader(loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                labels = labels.to(device, dtype=torch.long)
                preds = model(imgs).argmax(1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return 100. * correct / total

    val_acc  = eval_loader(val_loader)
    test_acc = eval_loader(test_loader)
    print(f"Validation Accuracy after epoch 0: {val_acc:.2f}%")
    print(f"Test Accuracy after epoch 0: {test_acc:.2f}%\n")
    val_acc_list.append(val_acc); test_acc_list.append(test_acc)
    train_acc_list.append(float(100/29))      # random baseline

    start_time = time.time()

    
    for epoch in range(num_epochs):

        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for step, (imgs, labels) in enumerate(train_loader, start=1):
            imgs, labels = imgs.to(device), labels.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total   += labels.size(0)

            if step % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                      f"Step [{step}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        
        if epoch < warm_epochs:
            warmup.step()
        else:
            cosine.step()

        
        avg_loss   = running_loss / len(train_loader)
        train_acc  = 100. * correct / total
        val_acc    = eval_loader(val_loader)
        test_acc   = eval_loader(test_loader)

        loss_list.append(avg_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Loss: {avg_loss:.4f}  "
              f"Train: {train_acc:.2f}%  "
              f"Val: {val_acc:.2f}%  "
              f"Test: {test_acc:.2f}%")

        
        if early_stopping:
            if val_acc > best_val_acc:
                best_val_acc = val_acc; no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("Early stopping: no val‑improvement "
                          f"for {patience} epochs.")
                    break
    

    total_time = time.time() - start_time
    print(f"Total training time: {total_time/60:.1f} min")

    return {
        "loss": loss_list,
        "train_accuracy": train_acc_list,
        "val_accuracy": val_acc_list,
        "test_accuracy": test_acc_list,
        "training_time": total_time
    }



def train_AMSGrad(model, 
                  train_loader, 
                  val_loader, 
                  test_loader, 
                  device, 
                  num_epochs=100, 
                  lr=0.001, 
                  weight_decay=0.0):

    loss_list = []
    train_accuracy_list = []
    val_accuracy_list = []
    test_accuracy_list = []

    def eval_loader(loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                labels = labels.to(device, dtype=torch.long)
                preds = model(imgs).argmax(1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return 100. * correct / total
    
    val_acc  = eval_loader(val_loader)
    test_acc = eval_loader(test_loader)
    print(f"Validation Accuracy after epoch 0: {val_acc:.2f}%")
    print(f"Test Accuracy after epoch 0: {test_acc:.2f}%\n")
    val_accuracy_list.append(val_acc); test_accuracy_list.append(test_acc)
    train_accuracy_list.append(float(100/29))      # random baseline

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)

    # Start timing the training process
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = running_loss / len(train_loader)
        loss_list.append(avg_loss)
        train_accuracy = 100 * correct / total
        train_accuracy_list.append(train_accuracy)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Average Loss: {avg_loss:.4f}")

        avg_loss = running_loss / len(train_loader)
        loss_list.append(avg_loss)
        train_accuracy = 100 * correct / total
        train_accuracy_list.append(train_accuracy)

        val_acc    = eval_loader(val_loader)
        test_acc   = eval_loader(test_loader)

        loss_list.append(avg_loss)
        val_accuracy_list.append(val_acc)
        test_accuracy_list.append(test_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Loss: {avg_loss:.4f}  "
              f"Train: {train_accuracy:.2f}%  "
              f"Val: {val_acc:.2f}%  "
              f"Test: {test_acc:.2f}%")

    total_training_time = time.time() - start_time
    print(f"Total training time: {total_training_time:.2f} seconds")

    metrics = {
        "loss": loss_list,
        "train_accuracy": train_accuracy_list,
        "val_accuracy": val_accuracy_list,
        "test_accuracy": test_accuracy_list,
        "training_time": total_training_time
    }
    return metrics

def run_first_grid_search_maornet(device):

    num_epochs = 80
    momentum = 0.9
    weight_decay = 0.0005


    gaussian_initialization = [True, False]
    batch_size_options = [128, 256, 512]
    learning_rate_options = [0.01, 0.001, 0.0001]
    seeds = [0, 1, 2]

    csv_file = os.path.join("dataset", "MAMe_dataset.csv")
    img_dir = os.path.join("dataset", "imgs")

    # Precomputed normalization values
    norm_mean = [0.6191, 0.5868, 0.5411]
    norm_std = [0.1793, 0.1792, 0.1802]
    

    crop_size = (224, 224)

    transform_train = transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    transform_val_test = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    results = []
    backup_file = "experiment_results_first_grid_search_maronet.json"

    for gaussian_init in gaussian_initialization:
        for batch_size in batch_size_options:
            for lr in learning_rate_options:
                for seed in seeds:

                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(seed)
                    
                    train_dataset = MAMeDataset(csv_file=csv_file, img_dir=img_dir,
                                                transform=transform_train, subset='train')
                    val_dataset = MAMeDataset(csv_file=csv_file, img_dir=img_dir,
                                              transform=transform_val_test, subset='val')
                    test_dataset = MAMeDataset(csv_file=csv_file, img_dir=img_dir,
                                               transform=transform_val_test, subset='test')

                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
                    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

                    model = MaorNet(num_classes=29, gaussian_initialization=gaussian_init)
                    model.to(device)

                    config_id = (f"MaorNet_bs_{batch_size}_lr_{lr}_seed_{seed}_"
                                 f"gaussian_{gaussian_init}_epochs_{num_epochs}_"
                                 f"mom_{momentum}_wd_{weight_decay}")
                    print("Running config:", config_id)

                    metrics = train_sgd(model, train_loader, val_loader, test_loader, device,
                                    num_epochs=num_epochs, lr=lr,
                                    momentum=momentum, weight_decay=weight_decay)
                    
                    result_entry = {
                        "config_id": config_id,
                        "model": "MaorNet",
                        "hyperparameters": {
                            "gaussian_initialization": gaussian_init,
                            "batch_size": batch_size,
                            "learning_rate": lr,
                            "seed": seed,
                            "num_epochs": num_epochs,
                            "momentum": momentum,
                            "weight_decay": weight_decay,
                            "batch_norm": True,
                            "dropout": True
                        },
                        "metrics": metrics
                    }
                    results.append(result_entry)

                    try:
                        with open(backup_file, "w") as f:
                            json.dump(results, f)
                    except Exception as e:
                        print("Error saving backup:", e)

                    del model
                    torch.cuda.empty_cache()

    return results

def run_hyperparameters_experiments_maornet(device):

    config_list = [
        {"learning_rate": 0.001, "momentum": 0.0, "weight_decay": 0.0005, "dropout": True, "batch_norm": True}, # No momentum
        {"learning_rate": 0.001, "momentum": 0.9, "weight_decay": 0.0, "dropout": True, "batch_norm": True}, # No weight decay
        {"learning_rate": 0.001, "momentum": 0.9, "weight_decay": 0.0005, "dropout": False, "batch_norm": True}, # No dropout
        {"learning_rate": 0.001, "momentum": 0.9, "weight_decay": 0.0005, "dropout": True, "batch_norm": False}, # No batch norm
        {"learning_rate": 1.0, "momentum": 0.9, "weight_decay": 0.0005, "dropout": True, "batch_norm": True}, # Big LR
        {"learning_rate": 0.001, "momentum": 0.9, "weight_decay": 1.0, "dropout": True, "batch_norm": True}, # Big WD
        {"learning_rate": 0.000001, "momentum": 0.9, "weight_decay": 0.0005, "dropout": True, "batch_norm": True}, # small LR
        {"learning_rate": 0.001, "momentum": 0.9, "weight_decay": 0.0, "dropout": False, "batch_norm": True}, # No regularization
        {"learning_rate": 0.001, "momentum": 0.9, "weight_decay": 0.0, "dropout": False, "batch_norm": False}, # No regularization + No batch norm
    ]

    seeds=[0, 1, 2]
    

    num_epochs = 80
    batch_size = 128
    gaussian_init = True
    
    csv_file = os.path.join("dataset", "MAMe_dataset.csv")
    img_dir = os.path.join("dataset", "imgs")
    norm_mean = [0.6191, 0.5868, 0.5411]
    norm_std = [0.1793, 0.1792, 0.1802]
    crop_size = (224, 224)
    
    transform_train = transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    transform_val_test = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    backup_file = "experiment_results_config_list_maornet.json"
    results = []

    for config in config_list:
        lr = config.get("learning_rate")
        momentum = config.get("momentum")
        weight_decay = config.get("weight_decay")
        dropout = config.get("dropout")
        batch_norm = config.get("batch_norm")
        
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            
            train_dataset = MAMeDataset(csv_file=csv_file, img_dir=img_dir,
                                        transform=transform_train, subset='train')
            val_dataset = MAMeDataset(csv_file=csv_file, img_dir=img_dir,
                                      transform=transform_val_test, subset='val')
            test_dataset = MAMeDataset(csv_file=csv_file, img_dir=img_dir,
                                       transform=transform_val_test, subset='test')
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
            
            model = MaorNet(num_classes=29, gaussian_initialization=gaussian_init, dropout=dropout, batch_norm=batch_norm)
            model.to(device)
            
            config_id = (f"MaorNet_bs_{batch_size}_lr_{lr}_seed_{seed}_"
                         f"gaussian_{gaussian_init}_epochs_{num_epochs}_"
                         f"mom_{momentum}_wd_{weight_decay}_dropout_{dropout}_batchnorm_{batch_norm}")
            print("Running config:", config_id)
            
            metrics = train_sgd(model, train_loader, val_loader, test_loader, device,
                            num_epochs=num_epochs, lr=lr,
                            momentum=momentum, weight_decay=weight_decay)
            
            result_entry = {
                "config_id": config_id,
                "hyperparameters": {
                    "seed": seed,
                    "gaussian_initialization": gaussian_init,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "num_epochs": num_epochs,
                    "momentum": momentum,
                    "weight_decay": weight_decay,
                    "dropout": dropout,
                    "batch_norm": batch_norm
                },
                "metrics": metrics
            }
            results.append(result_entry)
            
            try:
                with open(backup_file, "w") as f:
                    json.dump(results, f)
            except Exception as e:
                print("Error saving backup:", e)
            
            del model
            torch.cuda.empty_cache()
    
    return results


def run_optimal_experiments_maornet(device):

    dropout_rate_list = [0.3, 0.4, 0.5, 0.6]
    weight_decay_list = [0.0001, 0.0005, 0.001]

    seeds=[0, 1, 2]
    
    learning_rate = 0.001
    momentum = 0.9
    num_epochs = 120
    batch_size = 128
    gaussian_init = False
    batch_norm = True
    dropout = True

    lr_scheduling = True
    early_stopping = True
    
    csv_file = os.path.join("dataset", "MAMe_dataset.csv")
    img_dir = os.path.join("dataset", "imgs")
    norm_mean = [0.6191, 0.5868, 0.5411]
    norm_std = [0.1793, 0.1792, 0.1802]
    crop_size = (224, 224)
    
    transform_train = transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    transform_val_test = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    backup_file = "experiment_optimal_results_config_list_maornet.json"
    results = []

    for dropout_rate in dropout_rate_list:
        for weight_decay in weight_decay_list:
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                
                train_dataset = MAMeDataset(csv_file=csv_file, img_dir=img_dir,
                                            transform=transform_train, subset='train')
                val_dataset = MAMeDataset(csv_file=csv_file, img_dir=img_dir,
                                        transform=transform_val_test, subset='val')
                test_dataset = MAMeDataset(csv_file=csv_file, img_dir=img_dir,
                                        transform=transform_val_test, subset='test')
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
                
                model = MaorNet(num_classes=29, gaussian_initialization=gaussian_init, dropout=dropout, batch_norm=batch_norm, dropout_rate=dropout_rate)
                model.to(device)
                
                config_id = (f"MaorNet_bs_{batch_size}_lr_{learning_rate}_seed_{seed}_"
                            f"gaussian_{gaussian_init}_epochs_{num_epochs}_"
                            f"mom_{momentum}_wd_{weight_decay}_dropout_rate_{dropout_rate}_batchnorm_{batch_norm}")
                print("Running config:", config_id)
                
                metrics = train_sgd(model, train_loader, val_loader, test_loader, device,
                                num_epochs=num_epochs, lr=learning_rate,
                                momentum=momentum, weight_decay=weight_decay,
                                lr_scheduling=lr_scheduling, early_stopping=early_stopping)
                
                result_entry = {
                    "config_id": config_id,
                    "hyperparameters": {
                        "seed": seed,
                        "gaussian_initialization": gaussian_init,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "num_epochs": num_epochs,
                        "momentum": momentum,
                        "weight_decay": weight_decay,
                        "dropout_rate": dropout_rate,
                        "batch_norm": batch_norm
                    },
                    "metrics": metrics
                }
                results.append(result_entry)
                
                try:
                    with open(backup_file, "w") as f:
                        json.dump(results, f)
                except Exception as e:
                    print("Error saving backup:", e)
                
                del model
                torch.cuda.empty_cache()
    
    return results


def run_first_grid_search_resnet18(device):
    num_epochs = 80
    momentum = 0.9
    weight_decay = 0.0005

    gaussian_initialization = [True, False]
    batch_size_options = [128, 256, 512]
    learning_rate_options = [0.01, 0.001, 0.0001]
    seeds = [0, 1, 2]

    csv_file = os.path.join("dataset", "MAMe_dataset.csv")
    img_dir = os.path.join("dataset", "imgs")

    # Precomputed normalization values
    norm_mean = [0.6191, 0.5868, 0.5411]
    norm_std = [0.1793, 0.1792, 0.1802]
    

    crop_size = (224, 224)

    transform_train = transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    transform_val_test = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    results = []
    backup_file = "experiment_results_first_grid_search_resnet18.json"

    for gaussian_init in gaussian_initialization:
        for batch_size in batch_size_options:
            for lr in learning_rate_options:
                for seed in seeds:

                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(seed)
                    
                    train_dataset = MAMeDataset(csv_file=csv_file, img_dir=img_dir,
                                                transform=transform_train, subset='train')
                    val_dataset = MAMeDataset(csv_file=csv_file, img_dir=img_dir,
                                              transform=transform_val_test, subset='val')
                    test_dataset = MAMeDataset(csv_file=csv_file, img_dir=img_dir,
                                               transform=transform_val_test, subset='test')

                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
                    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

                    model = ResNet18(num_classes=29, gaussian_initialization=gaussian_init, p_head=0.0, p_block=0.0)
                    model.to(device)

                    config_id = (f"ResNet18_bs_{batch_size}_lr_{lr}_seed_{seed}_"
                                 f"gaussian_{gaussian_init}_epochs_{num_epochs}_"
                                 f"wd_{weight_decay}")
                    print("Running config:", config_id)

                    metrics = train_sgd(model, train_loader, val_loader, test_loader, device,
                                    num_epochs=num_epochs, lr=lr, momentum=momentum, weight_decay=weight_decay)

                    result_entry = {
                        "config_id": config_id,
                        "model": "Resnet18",
                        "hyperparameters": {
                            "gaussian_initialization": gaussian_init,
                            "batch_size": batch_size,
                            "learning_rate": lr,
                            "seed": seed,
                            "num_epochs": num_epochs,
                            "weight_decay": weight_decay,
                            "batch_norm": True,
                            "dropout": False
                        },
                        "metrics": metrics
                    }
                    results.append(result_entry)

                    try:
                        with open(backup_file, "w") as f:
                            json.dump(results, f)
                    except Exception as e:
                        print("Error saving backup:", e)

                    del model
                    torch.cuda.empty_cache()

    return results

def run_hyperparameters_experiments_resnet18(device):

    config_list = [
        {"learning_rate": 0.01, "momentum": 0.0, "weight_decay": 0.0005, "dropout": True, "batch_norm": True}, # No momentum
        {"learning_rate": 0.01, "momentum": 0.9, "weight_decay": 0.0, "dropout": True, "batch_norm": True}, # No weight decay
        {"learning_rate": 0.01, "momentum": 0.9, "weight_decay": 0.0005, "dropout": False, "batch_norm": True}, # No dropout
        {"learning_rate": 0.01, "momentum": 0.9, "weight_decay": 0.0005, "dropout": True, "batch_norm": False}, # No batch norm
        {"learning_rate": 1.0, "momentum": 0.9, "weight_decay": 0.0005, "dropout": True, "batch_norm": True}, # Big LR
        {"learning_rate": 0.01, "momentum": 0.9, "weight_decay": 1.0, "dropout": True, "batch_norm": True}, # Big WD
        {"learning_rate": 0.000001, "momentum": 0.9, "weight_decay": 0.0005, "dropout": True, "batch_norm": True}, # small LR
        {"learning_rate": 0.01, "momentum": 0.9, "weight_decay": 0.0, "dropout": False, "batch_norm": True}, # No regularization
        {"learning_rate": 0.01, "momentum": 0.9, "weight_decay": 0.0, "dropout": False, "batch_norm": False}, # No regularization + No batch norm
    ]

    seeds=[0, 1, 2]
    

    num_epochs = 80
    batch_size = 128
    gaussian_init = False
    
    csv_file = os.path.join("dataset", "MAMe_dataset.csv")
    img_dir = os.path.join("dataset", "imgs")
    norm_mean = [0.6191, 0.5868, 0.5411]
    norm_std = [0.1793, 0.1792, 0.1802]
    crop_size = (224, 224)
    
    transform_train = transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    transform_val_test = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    backup_file = "hyperparameter_experiments_resnet18.json"
    results = []

    for config in config_list:
        lr = config.get("learning_rate")
        momentum = config.get("momentum")
        weight_decay = config.get("weight_decay")
        dropout = config.get("dropout")
        batch_norm = config.get("batch_norm")
        
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            
            train_dataset = MAMeDataset(csv_file=csv_file, img_dir=img_dir,
                                        transform=transform_train, subset='train')
            val_dataset = MAMeDataset(csv_file=csv_file, img_dir=img_dir,
                                      transform=transform_val_test, subset='val')
            test_dataset = MAMeDataset(csv_file=csv_file, img_dir=img_dir,
                                       transform=transform_val_test, subset='test')
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)
            
            if dropout:
                p_head = 0.5
                p_block = 0.5
            else:
                p_head, p_block = 0.0, 0.0

            model = ResNet18(num_classes=29, gaussian_initialization=gaussian_init, batch_norm=batch_norm, p_head=p_head, p_block=p_block)
            model.to(device)
            
            config_id = (f"ResNet18_bs_{batch_size}_lr_{lr}_seed_{seed}_"
                         f"gaussian_{gaussian_init}_epochs_{num_epochs}_"
                         f"mom_{momentum}_wd_{weight_decay}_dropout_{dropout}_batchnorm_{batch_norm}")
            print("Running config:", config_id)
            
            metrics = train_sgd(model, train_loader, val_loader, test_loader, device,
                            num_epochs=num_epochs, lr=lr,
                            momentum=momentum, weight_decay=weight_decay)
            
            # Record configuration and results
            result_entry = {
                "config_id": config_id,
                "hyperparameters": {
                    "seed": seed,
                    "gaussian_initialization": gaussian_init,
                    "batch_size": batch_size,
                    "learning_rate": lr,
                    "num_epochs": num_epochs,
                    "momentum": momentum,
                    "weight_decay": weight_decay,
                    "dropout": dropout,
                    "batch_norm": batch_norm
                },
                "metrics": metrics
            }
            results.append(result_entry)
            
            try:
                with open(backup_file, "w") as f:
                    json.dump(results, f)
            except Exception as e:
                print("Error saving backup:", e)
            
            del model
            torch.cuda.empty_cache()
    
    return results

def run_optimal_resnet18(device):

    num_epochs = 200
    weight_decay = 0.0001
    momentum = 0.9

    seeds = [0, 1, 2]

    lr = 0.05 * (128 / 256)

    gaussian_init = True
    batch_size = 128

    csv_file = os.path.join("dataset", "MAMe_dataset.csv")
    img_dir = os.path.join("dataset", "imgs")

    # Precomputed normalization values
    norm_mean = [0.6191, 0.5868, 0.5411]
    norm_std = [0.1793, 0.1792, 0.1802]
    

    crop_size = (224, 224)

    transform_train = transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    transform_val_test = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    results = []
    backup_file = "experiment_resnet18_optimal.json"

    for seed in seeds:

        torch.manual_seed(seed)
        np.random.seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        
        train_dataset = MAMeDataset(csv_file=csv_file, img_dir=img_dir,
                                    transform=transform_train, subset='train')
        val_dataset = MAMeDataset(csv_file=csv_file, img_dir=img_dir,
                                    transform=transform_val_test, subset='val')
        test_dataset = MAMeDataset(csv_file=csv_file, img_dir=img_dir,
                                    transform=transform_val_test, subset='test')

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

        model = ResNet18(num_classes=29, gaussian_initialization=gaussian_init)
        model.to(device)

        config_id = (f"ResNet18_bs_{batch_size}_lr_{lr}_seed_{seed}_"
                        f"gaussian_{gaussian_init}_epochs_{num_epochs}_"
                        f"wd_{weight_decay}")
        print("Running config:", config_id)

        metrics = train_sgd_no_overfit(model, train_loader, val_loader, test_loader, device,
                        num_epochs=num_epochs, lr=lr, momentum=momentum, weight_decay=weight_decay)

        result_entry = {
            "config_id": config_id,
            "model": "Resnet18",
            "hyperparameters": {
                "gaussian_initialization": gaussian_init,
                "batch_size": batch_size,
                "learning_rate": lr,
                "seed": seed,
                "num_epochs": num_epochs,
                "weight_decay": weight_decay,
                "batch_norm": True,
                "dropout": True
            },
            "metrics": metrics
        }
        results.append(result_entry)

        try:
            with open(backup_file, "w") as f:
                json.dump(results, f)
        except Exception as e:
            print("Error saving backup:", e)

        del model
        torch.cuda.empty_cache()

    return results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    compute_normalization_values()
    
    run_first_grid_search_maornet(device)
    run_hyperparameters_experiments_maornet(device)
    run_optimal_experiments_maornet(device)

    run_first_grid_search_resnet18(device)
    run_hyperparameters_experiments_resnet18(device)
    run_optimal_resnet18(device)
    
    print("\nCompleted!")