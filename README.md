# ViT-Base Deepfake Image Detection System

A compact, end-to-end pipeline for detecting AI-generated (deepfake) images using a fine-tuned Vision Transformer (ViT-Base). Includes preprocessing, augmentation, training, evaluation and batch inference for binary classification (REAL / FAKE).

---

## Quick summary
- Model: `google/vit-base-patch16-224` backbone + custom classifier head (≈86M params, ~330MB).  
- Tasks: resize & preprocess images, optional deblur/sharpen, augment, train with Improved Focal Loss, evaluate, and export JSON predictions.  
- Dataset: NOT included in repo — you must supply and organize local data (instructions below).

---

## Table of contents
- [Requirements](#requirements)  
- [Installation](#installation)  
- [Dataset (Not included)](#dataset-not-included)  
- [Preprocessing & Augmentation](#preprocessing--augmentation)  
- [Training](#training)  
- [Evaluation](#evaluation)  
- [Inference](#inference)  
- [Results (example)](#results-example)  
- [Troubleshooting](#troubleshooting)  
- [License](#license)

---

## Requirements
Recommended: Python 3.8+, GPU with 8GB+ VRAM.  
Key packages:
- torch, torchvision, transformers, pillow, scikit-learn, matplotlib, seaborn, tqdm, opencv-python, numpy

Install example:
```bash
# install core dependencies (adjust torch command per CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers pillow scikit-learn matplotlib seaborn tqdm opencv-python numpy
```

---

## Installation
```bash
git clone https://github.com/Asmit-ctrl/DeepFake-Image-detection-Challenge-Synergy--IIITB-.git
cd DeepFake-Image-detection-Challenge-Synergy--IIITB-
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
# source venv/bin/activate
pip install -r requirements.txt   # optional: create from listed libs
```

---

## Dataset (provided by IIITB)
This project was made using the dataset provided by the challenge organisers i.e. SYNERGY (IIITB).
link:- https://drive.google.com/drive/folders/1kR1qLW9DHkSBzBAIViX7G5hBtKVT8Uqp (link may not work in future)
Recommended sources:
- FaceForensics++ — https://github.com/ondyari/FaceForensics  
- Celeb-DF — https://github.com/yuezunli/celeb-deepfakeforensics  
- DFDC — https://ai.facebook.com/datasets/dfdc/  
- 140k Real and Fake Faces (Kaggle)

Organize local dataset like:
```
deepfake-dataset/
    real/
    fake/
    real_val/
    fake_val/
    test/           # unlabeled images for inference
```
Update paths inside the notebook/script (Cell 11):
```python
real_train_dir = 'C:/path/to/dataset/real'
fake_train_dir = 'C:/path/to/dataset/fake'
real_val_dir   = 'C:/path/to/dataset/real_val'
fake_val_dir   = 'C:/path/to/dataset/fake_val'
```

---

## Preprocessing & Augmentation
- Resize to 224×224 (PIL LANCZOS), convert to RGB.
- Optional: Gaussian blur + SHARPEN.
- Augmentations available: horizontal/vertical flips, 90° rotation, crop+zoom.

Example resize usage (already in notebook):
```python
from PIL import Image
img = Image.open(path).convert("RGB")
img = img.resize((224,224), Image.Resampling.LANCZOS)
img.save(out_path)
```

---

## Training
- Loss: Improved Focal Loss (alpha, gamma configurable).
- Optimizer: AdamW with differential LRs (backbone lr/5, classifier lr).
- Scheduler: CosineAnnealingWarmRestarts.
- Regularization: Dropout, gradient clipping, early stopping (patience=10).
- Batch size used in example: 32 (adjust to GPU memory).

Run training cell / function:
```python
trainer, model_path = train_vit_base_separate_folders(
  real_train_dir, fake_train_dir, real_val_dir, fake_val_dir, num_epochs=20
)
```
Model checkpoints saved as `vit_base_best.pth` and `vit_base_final_<timestamp>.pth`.

---

## Evaluation
Built-in evaluation provides:
- Accuracy, precision, recall, F1
- Confusion matrix and classification report
- Plots: training/validation loss and accuracy

Example call:
```python
evaluate_vit_base(trainer.model, real_val_dir, fake_val_dir)
```

---

## Inference
Batch inference script returns JSON mapping filename (no extension) → probability(fake).

Usage:
```python
predictions = predict_test_images(
  model_path='C:/path/to/vit_base_best.pth',
  test_dir='C:/path/to/test',
  output_json='teamname_prediction.json'
)
```

Output example:
```json
{
  "image_0001": 0.6810,
  "image_0002": 0.0331,
  ...
}
```

---

## Results (example)
Reported validation performance from sample run:
- Best validation accuracy: 96.20%  
- Final validation accuracy: 93.57%  
- Confusion matrix example:
```
[[161, 10],
 [ 12,159]]
```
Your results will vary by dataset and hyperparameters.

---


---

## Quick checklist
- [ ] Clone repo  
- [ ] Install dependencies  
- [ ] Download and arrange dataset locally  
- [ ] Update dataset paths in notebook/script  
- [ ] Run preprocessing (optional augmentation)  
- [ ] Train and evaluate  
- [ ] Run inference and export JSON

--- 

For exact commands and code snippets, see the notebook cells in `Deepfake_Challenge_Synergy_IIITB.ipynb`.// filepath: README.md
