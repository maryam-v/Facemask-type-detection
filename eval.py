"""
eval.py
--------
Evaluate trained CNN models for Face Mask Classification.
Loads saved checkpoints, runs inference, and generates metrics/plots.
"""

import argparse
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from datasets import load_data, split_data, preprocess_data, MaskDataset
from models import NetVer1, NetVer2, NetVer3, SimpleCNN, ResNet18, MobileNetV2, CustomDeepCNN


def test(model, test_loader):
    """Run inference on test set and compute predictions/labels."""
    y_true, y_pred = [], []
    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    acc = (np.array(y_true) == np.array(y_pred)).mean() * 100
    print(f"Testing Accuracy: {acc:.2f}%")
    return y_true, y_pred


def plot_metrics(info_loss_acc, save_path):
    """Plot training/validation loss and accuracy curves."""
    # Loss
    plt.figure(figsize=(10, 7))
    plt.plot(info_loss_acc['train_loss'], 'c', linewidth=2, label='Training Loss')
    plt.plot(info_loss_acc['val_loss'], 'b', linewidth=2, label='Validation Loss')
    plt.grid()
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.tight_layout()
    plt.savefig(f"{save_path}/loss.jpg", dpi=300)

    # Accuracy
    plt.figure(figsize=(10, 7))
    plt.plot(info_loss_acc['train_acc'], 'c', linewidth=2, label='Training Accuracy')
    plt.plot(info_loss_acc['val_acc'], 'b', linewidth=2, label='Validation Accuracy')
    plt.grid()
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.tight_layout()
    plt.savefig(f"{save_path}/accuracy.jpg", dpi=300)


def plot_confusion_matrix(y_true, y_pred, categories, save_path):
    """Generate and save confusion matrix heatmap."""
    conf_mat = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", conf_mat)
    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=categories))

    df_cm = pd.DataFrame(conf_mat, categories, categories)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{save_path}/conf_mat.png", dpi=300)


def main(args):
    # Load and preprocess data
    data, labels = load_data()
    data_train, data_val, data_test, labels_train, labels_val, labels_test = split_data(data, labels)
    _, _, test_images, _, _, test_labels = preprocess_data(
        data_train, data_val, data_test, labels_train, labels_val, labels_test
    )

    test_dataset = MaskDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

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

    # Load weights
    model.load_state_dict(torch.load(f"Models/{args.model}/model_final.pth"))
    model.eval()

    # Load training history
    with open(f"Models/{args.model}/info_loss_acc.pkl", 'rb') as f:
        info_loss_acc = pickle.load(f)

    # Run test
    y_true, y_pred = test(model, test_loader)

    # Plot metrics
    plot_metrics(info_loss_acc, f"Figures/{args.model}")

    # Confusion matrix
    categories = ['Cloth Mask', 'Mask Worn Incorrectly', 'N-95 Mask', 'No Face Mask', 'Surgical Mask']
    plot_confusion_matrix(y_true, y_pred, categories, f"Figures/{args.model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CNN models for mask classification")
    parser.add_argument("--model", type=str, default="ver1", help="Model version: ver1, ver2, ver3")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for testing")
    args = parser.parse_args()

    main(args)
