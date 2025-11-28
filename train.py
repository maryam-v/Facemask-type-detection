"""
train.py
---------
Handles training of CNN models for Face Mask Classification.
Imports datasets and models, runs training loop, and saves checkpoints.
"""

import argparse
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import load_data, split_data, preprocess_data, MaskDataset
from models import NetVer1, NetVer2, NetVer3, SimpleCNN, ResNet18, MobileNetV2, CustomDeepCNN


def train_val(model, train_loader, val_loader, num_epochs, optimizer, criterion, save_path):
    """Train and validate a model, saving metrics and checkpoints."""
    loss_train_avg, loss_val_avg = [], []
    acc_train_avg, acc_val_avg = [], []

    info_loss_acc = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses, train_correct, train_total = [], 0, 0

        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc_epoch = train_correct / train_total
        loss_train_avg.append(sum(train_losses) / len(train_loader))
        acc_train_avg.append(train_acc_epoch)

        # Validation
        model.eval()
        val_losses, val_correct, val_total = [], 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc_epoch = val_correct / val_total
        loss_val_avg.append(sum(val_losses) / len(val_loader))
        acc_val_avg.append(val_acc_epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {loss_train_avg[-1]:.4f}, Train Acc: {train_acc_epoch*100:.2f}%, "
              f"Val Loss: {loss_val_avg[-1]:.4f}, Val Acc: {val_acc_epoch*100:.2f}%")

    # Save final model and metrics
    torch.save(model.state_dict(), f"{save_path}/model_final.pth")
    info_loss_acc['train_loss'] = loss_train_avg
    info_loss_acc['train_acc'] = acc_train_avg
    info_loss_acc['val_loss'] = loss_val_avg
    info_loss_acc['val_acc'] = acc_val_avg

    with open(f"{save_path}/info_loss_acc.pkl", 'wb') as f:
        pickle.dump(info_loss_acc, f)

    return model, info_loss_acc


def main(args):
    # Load and preprocess data
    data, labels = load_data()
    data_train, data_val, data_test, labels_train, labels_val, labels_test = split_data(data, labels)
    train_images, train_labels, val_images, val_labels, test_images, test_labels = preprocess_data(
        data_train, data_val, data_test, labels_train, labels_val, labels_test
    )

    # Create datasets and loaders
    train_dataset = MaskDataset(train_images, train_labels)
    val_dataset = MaskDataset(val_images, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)


    

    # Select model
    if args.model == "ver1":
        model = NetVer1()
    elif args.model == "ver2":
        model = NetVer2()
    elif args.model == "ver3":
        model = NetVer3()
    elif args.model == "simple":
        model = SimpleCNN()
    elif args.model == "resnet18":
        model = ResNet18(pretrained=True)
    elif args.model == "mobilenetv2":
        model = MobileNetV2(pretrained=True)
    elif args.model == "deepcnn":
        model = CustomDeepCNN()
    else:
        raise ValueError("Invalid model version. Choose from: ver1, ver2, ver3")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    train_val(model, train_loader, val_loader, args.epochs, optimizer, criterion, f"Models/{args.model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN models for mask classification")
    parser.add_argument("--model", type=str, default="ver1", help="Model version: ver1, ver2, ver3")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    main(args)
