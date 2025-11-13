# CIFAR-10 Image Classifier

This project implements a convolutional neural network (CNN) using TensorFlow/Keras to classify images from the CIFAR-10 dataset. It demonstrates a complete machine learning workflow: data preparation, model construction, training, evaluation, and visualization.

## Overview

The model is trained on the CIFAR-10 dataset, which contains 60,000 32x32 color images across 10 classes. The dataset is automatically downloaded and preprocessed (normalized and one-hot encoded) before training.

The workflow includes:
- Efficient data preprocessing and caching
- CNN model creation using Keras Sequential API
- Training with validation monitoring
- Saving the best model during training
- Visualization of accuracy and loss curves

## Model Summary

**Architecture**
- Two convolutional blocks with Batch Normalization, MaxPooling, and Dropout
- One fully connected Dense layer with ReLU activation
- Output layer with softmax activation for classification
- Optimizer: Adam  
- Loss: categorical_crossentropy  
- Metrics: accuracy

**Results**
- Baseline model achieves around 80% validation accuracy after 20 epochs
- With data augmentation and learning rate scheduling, the performance improves further

## Project Structure

image_classifier/
├── src/
│ ├── data_prep.py # Downloads and preprocesses CIFAR-10
│ ├── model.py # Builds, trains, and evaluates the CNN
│ └── ...
├── model_train.png # Training/validation accuracy plot
├── requirements.txt # Python dependencies
└── .gitignore

bash
Копировать код

## Setup and Usage

### 1. Clone the repository
```bash
git clone https://github.com/katr1nas/image_classifier_cifar10.git
cd image_classifier_cifar10
```
2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Train the model
```bash
cd src
python model.py
```
The script will:
1. Load or download CIFAR-10 data
2. Train the CNN model
3. Save the best model as best_model.h5
4. Display a plot of training and validation accuracy


Example Results
Baseline model (no augmentation):

Approximate validation accuracy: 80% after 20 epochs.

Future Improvements
1. Add data augmentation (rotation, translation, flips)
2. Implement learning rate scheduling
3. Experiment with deeper architectures (e.g., ResNet-style blocks)
4. Extend to CIFAR-100 or custom datasets
5. Package the model as a Flask or FastAPI API for inference

Author
Serghei Barladean
Focused on systematic skill development in machine learning, trading, and algorithmic strategy design.

License
This project is open for educational and research purposes. You may adapt and extend it freely while keeping proper attribution.
