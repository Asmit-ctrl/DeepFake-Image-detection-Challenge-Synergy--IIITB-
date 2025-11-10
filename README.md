ViT-Base Deepfake Image Detection System

A comprehensive deep learning pipeline for detecting AI-generated (deepfake) images using the Vision Transformer (ViT-Base) architecture. This project includes complete data preprocessing, augmentation, training, evaluation, and inference capabilities for binary classification of real vs. fake images.

📋 Table of Contents

Overview

Features

System Requirements

Installation

Project Structure

Dataset Preparation

Model Architecture

Training Pipeline

Evaluation

Inference

Results

Troubleshooting

License

🎯 Overview

This project implements a state-of-the-art deepfake detection system using Vision Transformer Base (ViT-Base) architecture. The system can:

Process and augment large-scale image datasets

Train a robust binary classifier to distinguish real from AI-generated images

Achieve 96%+ validation accuracy

Generate batch predictions on unseen test data

Run efficiently on consumer-grade GPUs (8GB+ VRAM)

The ViT-Base model contains 86 million parameters (~330MB) and leverages transfer learning from Google's pre-trained vit-base-patch16-224 checkpoint, fine-tuned specifically for deepfake detection.

✨ Features
Data Processing

Automated image resizing to 224×224 pixels with high-quality LANCZOS resampling

Color channel normalization (RGB conversion)

Advanced preprocessing: Gaussian blur simulation and sharpening filters

Extensive augmentation: horizontal/vertical flips, 90° rotation, crop+zoom (generates 4x more training data)

Model Capabilities

Pre-trained backbone: Leverages Google's ViT-Base trained on ImageNet-21k

Custom classifier head: 3-layer neural network optimized for binary classification

Improved Focal Loss: Handles class imbalance with configurable alpha/gamma parameters

Efficient architecture: Runs on RTX 3070/4060 or better

Training Features

Differential learning rates: Slower backbone updates (lr/5) vs. faster classifier updates

Advanced scheduling: Cosine annealing with warm restarts for optimal convergence

Regularization: Dropout (0.2-0.3), gradient clipping (max_norm=1.0), weight decay

Early stopping: Patience-based monitoring to prevent overfitting

Comprehensive logging: Real-time progress bars, loss/accuracy tracking, learning rate monitoring

Evaluation & Inference

Detailed metrics: Accuracy, precision, recall, F1-score, confusion matrices

Visualization: Training curves, accuracy plots, confusion matrix heatmaps

Batch inference: Process thousands of images efficiently

JSON output: Standard format for competition submissions

💻 System Requirements
Hardware

GPU: NVIDIA GPU with 8GB+ VRAM (tested on RTX A6000, works on RTX 3070/4060/4070)

RAM: 16GB+ recommended

Storage: 10GB+ (5GB for model checkpoints, 5GB+ for your dataset)

Software

OS: Windows 10/11, Linux, or macOS

Python: 3.8 or higher

CUDA: 11.7+ (for GPU acceleration)

Dependencies

Core libraries:

torch >= 2.0.0 - PyTorch deep learning framework

torchvision >= 0.15.0 - Computer vision utilities

transformers >= 4.30.0 - Hugging Face transformers (ViT models)

pillow >= 9.0.0 - Image loading and processing

scikit-learn >= 1.2.0 - Metrics and evaluation

matplotlib >= 3.5.0 - Plotting and visualization

seaborn >= 0.12.0 - Statistical visualizations

tqdm >= 4.65.0 - Progress bars

opencv-python >= 4.7.0 - Advanced image processing

numpy >= 1.24.0 - Numerical operations

📦 Installation
Step 1: Clone Repository
[https://github.com/Asmit-ctrl/DeepFake-Image-detection-Challenge-Synergy--IIITB-.git]


Step 2: Create Virtual Environment (Recommended)
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate

Step 3: Install Dependencies
# Install PyTorch (visit https://pytorch.org for your specific CUDA version)
# Example for CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install transformers pillow scikit-learn matplotlib seaborn tqdm opencv-python numpy

Step 4: Verify Installation
# Run in Python to verify GPU access
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

📊 Dataset Preparation
⚠️ Important: Dataset Not Included

This repository does not include any dataset. You must obtain and prepare your own deepfake detection dataset.

Recommended Dataset Sources

You can download datasets from these public sources:

FaceForensics++ (Recommended)

URL: https://github.com/ondyari/FaceForensics

Contains: 1000 real videos + 4000 manipulated videos

License: Research use only

Download: Requires academic email registration

Celeb-DF (v2)

URL: https://github.com/yuezunli/celeb-deepfakeforensics

Contains: 590 real + 5639 fake videos

License: Non-commercial use

Download: Google Drive link on GitHub

DFDC (Deepfake Detection Challenge)

URL: https://ai.facebook.com/datasets/dfdc/

Contains: ~100k videos

License: Research use

Download: Kaggle competition page

140k Real and Fake Faces

URL: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces

Contains: 70k real + 70k fake face images

License: Public domain

Download: Kaggle (requires account)

Custom Dataset

Collect your own real images (portraits, face photos)

Generate fake images using:

StyleGAN2/3

Stable Diffusion

Midjourney

DALL-E

Step 1: Download Dataset

Choose one of the sources above and download to your local machine.

Step 2: Organize Dataset Structure

After downloading, organize your images into this structure:

your-local-path/
└── deepfake-dataset/
    ├── train/
    │   ├── real/          # 60-70% of real images
    │   └── fake/          # 60-70% of fake images
    ├── val/
    │   ├── real/          # 15-20% of real images
    │   └── fake/          # 15-20% of fake images
    └── test/              # Remaining unlabeled images for inference


Recommended split:

Training: 70% (e.g., 3,316 real + 3,316 fake)

Validation: 15% (e.g., 171 real + 171 fake)

Testing: 15% (e.g., 500+ unlabeled images)

Step 3: Update Paths in Notebook

Open Untitled.ipynb and update these paths in Cell 11:

# UPDATE THESE PATHS TO YOUR LOCAL DATASET
real_train_dir = 'C:/path/to/your/dataset/train/real'
fake_train_dir = 'C:/path/to/your/dataset/train/fake'
real_val_dir = 'C:/path/to/your/dataset/val/real'
fake_val_dir = 'C:/path/to/your/dataset/val/fake'


And for inference (final cell):

# UPDATE THESE PATHS
MODEL_PATH = "C:/path/to/your/models/vit_base_best.pth"
TEST_DIR = "C:/path/to/your/dataset/test"
OUTPUT_JSON = "teamname_prediction.json"

Step 4: Image Preprocessing

The preprocessing pipeline handles resizing, normalization, and optional enhancement.

🧠 Model Architecture

The system uses the ViT-Base model with a custom classifier head for binary classification. The architecture includes:

Pre-trained ViT-Base: 86 million parameters, trained on ImageNet-21k.

Custom Classifier Head: A fully connected neural network consisting of three layers.

Activation: ReLU activation in the classifier layers.

Loss Function: Focal Loss (to address class imbalance).

🚀 Training Pipeline

The training process involves:

Data Loading: Use DataLoader for efficient mini-batch processing.

Training: Uses AdamW optimizer with learning rate scheduling and weight decay.

Regularization: Dropout layers, gradient clipping, early stopping.

Monitoring: Visualizes loss/accuracy curves using matplotlib and tensorboard.

Run the training pipeline:

train(model, train_loader, val_loader, epochs=20)

📊 Evaluation

After training, evaluate the model on the validation set to track metrics such as:

Accuracy: Overall classification performance.

Precision, Recall: For individual class (real/fake).

F1-Score: Harmonic mean of precision and recall.

Confusion Matrix: Visualize the true positives, true negatives, false positives, and false negatives.

Classification Report:

Classification Report:
              precision    recall  f1-score   support
        Real       0.93      0.94      0.94       171
        Fake       0.94      0.93      0.94       171
    accuracy                           0.94       342
   macro avg       0.94      0.94      0.94       342
weighted avg       0.94      0.94      0.94       342


Accuracy: 94%

Precision: 93% (Real), 94% (Fake)

Recall: 94% (Real), 93% (Fake)

F1-Score: 94% for both classes

🔮 Inference

For inference, the model will generate predictions and store them in a JSON file.

inference(model, test_loader, output_file='predictions.json')

📈 Results

The trained model achieves 94%+ accuracy on the validation set, demonstrating high reliability in distinguishing between real and fake images.
