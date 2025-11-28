"""
inference.py
-------------
Run inference on new sample images using trained CNN models
for Face Mask Classification.
"""

import argparse
import os
from PIL import Image
import torch
import torchvision.transforms as transforms

from models import NetVer1, NetVer2, NetVer3, SimpleCNN, ResNet18, MobileNetV2, CustomDeepCNN

# Categories mapping
CATEGORIES = {0: 'Cloth Mask',
              1: 'Mask Worn Incorrectly',
              2: 'N-95 Mask',
              3: 'No Face Mask',
              4: 'Surgical Mask'}

# ImageNet normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


def load_images(path):
    """Load and preprocess images from a directory."""
    images, filenames = [], []
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        try:
            image = Image.open(img_path)
            images.append(test_transforms(image))
            filenames.append(img)
        except Exception:
            pass
    return torch.stack(images), filenames


def run_inference(model, images_tensor):
    """Run inference and return predictions."""
    model.eval()
    preds = []
    with torch.no_grad():
        outputs = model(images_tensor)
        _, predicted = torch.max(outputs.data, 1)
        preds = predicted.tolist()
    return preds


def main(args):
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

    # Load sample images
    images_tensor, filenames = load_images(args.data)

    # Run inference
    preds = run_inference(model, images_tensor)

    # Print results
    for fname, pred in zip(filenames, preds):
        print(f"{fname}: {CATEGORIES[pred]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on sample images")
    parser.add_argument("--model", type=str, default="ver1", help="Model version: ver1, ver2, ver3")
    parser.add_argument("--data", type=str, default="SampleData/", help="Path to sample images")
    args = parser.parse_args()

    main(args)
