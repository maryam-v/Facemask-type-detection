# Face Mask Type Detection 🩺😷

A PyTorch pipeline for training, evaluating, and running inference on face mask classification using custom CNN architectures.

---

## 📂 Project Structure

```

MaskVision/ 
│── datasets.py # Data loading, preprocessing, and dataset classes 
│── models.py # CNN architectures (NetVer1, NetVer2, NetVer3) 
│── train.py # Training loop with argparse support 
│── eval.py # Evaluation, metrics, and plots 
│── inference.py # Run predictions on new sample images 
│── utils.py # Helper functions (visualization, reproducibility, checkpoints) 
│── Data/ # Dataset (organized by categories) 
│── SampleData/ # Demo images for inference 
│── requirements.txt # Dependencies 
│── README.md # Project overview

```
---

## 🚀 Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MaskVision.git
   cd MaskVision
   ```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Organize your dataset:

```
Data/
├── Cloth mask/
├── Mask worn incorrectly/
├── N-95_Mask/
├── No Face Mask/
└── Surgical Mask/
```


## 🏋️ Training
Train a model with:

```bash
python train.py --model ver1 --epochs 100 --batch_size 64 --lr 0.001
```

Options:

- --model → ver1, ver2, or ver3

- --epochs → number of training epochs

- --batch_size → batch size

- --lr → learning rate

Models and metrics are saved in Models/<version>/.

## 📊 Evaluation
Evaluate a trained model:

```bash
python eval.py --model ver1 --batch_size 64
```
Outputs:

Test accuracy

Classification report

Confusion matrix (Figures/<version>/conf_mat.png)

Loss/accuracy plots (Figures/<version>/loss.jpg, Figures/<version>/accuracy.jpg)

## 🔎 Inference
Run predictions on new sample images:

```bash
python inference.py --model ver1 --data SampleData/
```

Example output:

```bash
image1.jpg: Cloth Mask
image2.jpg: No Face Mask
```

## 📈 Results
- ver1 → 3‑layer CNN

- ver2 → 2‑layer CNN

- ver3 → 2‑layer CNN without pooling

- simple → lightweight baseline

- resnet18 → pretrained ResNet18 fine‑tuned for 5 classes

- mobilenetv2 → pretrained MobileNetV2 fine‑tuned for 5 classes

- deepcnn → deeper custom CNN
