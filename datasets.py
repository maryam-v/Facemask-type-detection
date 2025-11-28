"""
datasets.py
------------
Handles data loading, preprocessing, and PyTorch Dataset creation
for the Face Mask Classification project.
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Dataset directory and categories
DIRECTORY = 'Data/'
CATEGORIES = ['Cloth mask', 'Mask worn incorrectly', 'N-95_Mask', 'No Face Mask', 'Surgical Mask']

# ImageNet normalization standards
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


def load_data():
    """Load images and labels from directory."""
    data, labels = [], []
    for category in CATEGORIES:
        path = os.path.join(DIRECTORY, category)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            try:
                image = Image.open(img_path)
                data.append(image)
                labels.append(category)
            except Exception:
                pass
    return data, labels


def split_data(data, labels, train_size=1590, val_ratio=0.1):
    """Split dataset into train, validation, and test sets."""
    total_images = len(data)
    test_size = total_images - train_size
    test_ratio = test_size / total_images

    data_train, data_test, labels_train, labels_test = train_test_split(
        data, labels, test_size=test_ratio, random_state=42
    )
    data_train, data_val, labels_train, labels_val = train_test_split(
        data_train, labels_train, test_size=val_ratio, random_state=42
    )

    return data_train, data_val, data_test, labels_train, labels_val, labels_test


def preprocess_data(data_train, data_val, data_test, labels_train, labels_val, labels_test):
    """Apply transforms and convert labels to integers."""
    lb_make = LabelEncoder()
    labels_train = lb_make.fit_transform(labels_train)
    labels_val = lb_make.transform(labels_val)
    labels_test = lb_make.transform(labels_test)

    train_images = [train_transforms(img) for img in data_train]
    val_images = [test_transforms(img) for img in data_val]
    test_images = [test_transforms(img) for img in data_test]

    train_labels = torch.tensor(labels_train, dtype=torch.long)
    val_labels = torch.tensor(labels_val, dtype=torch.long)
    test_labels = torch.tensor(labels_test, dtype=torch.long)

    train_images_tensor = torch.stack(train_images)
    val_images_tensor = torch.stack(val_images)
    test_images_tensor = torch.stack(test_images)

    return (train_images_tensor, train_labels,
            val_images_tensor, val_labels,
            test_images_tensor, test_labels)


class MaskDataset(Dataset):
    """Custom Dataset for mask classification."""

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.labels)
