# 🔢 MNIST Handwritten Digit Classification

A deep learning project comparing three approaches to classifying handwritten digits from the MNIST dataset — a Convolutional Neural Network (CNN), a Feedforward Neural Network (FFNN), and a classical Support Vector Machine (SVM). Built to evaluate how well each architecture captures spatial patterns in image data.

## Overview
The MNIST dataset contains 70,000 grayscale images of handwritten digits (0–9), each 28x28 pixels. The goal is to train and compare models that can accurately classify these digits despite the variability in handwriting styles.

The project covers:

* **Data Preprocessing** — reshaping, normalization, and data augmentation (rotation, zoom)
* **Model Building** — CNN, FFNN, and SVM
* **Training & Tuning** — iterative experiments with layers, filters, and hyperparameters
* **Evaluation** — accuracy, precision, recall, F1-score, and confusion matrices
* **Visualization** — confusion matrix heatmaps and sample predictions per model

## Dataset
MNIST — 60,000 training images and 10,000 test images, loaded directly via:

```python
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
```

No external file is needed; Keras downloads the dataset automatically on first run.

## Results

| Model | Test Accuracy |
|---|---|
| **CNN** | **99.26%** |
| FFNN | 97.68% |
| SVM (linear kernel) | 94.04% |

* **CNN** performed best, learning spatial features through convolution and pooling layers, with very few misclassifications (131 out of 10,000).
* **FFNN** treats each pixel independently and showed more confusion between visually similar digits (e.g., 4 and 9), with some signs of overfitting.
* **SVM** used a linear kernel on flattened pixel vectors — a solid classical baseline, but with noticeably more confusion between similar-shaped digits since it can't learn hierarchical features.

## Tech Stack

* **Language**: Python
* **Libraries**: `tensorflow` / `keras`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn`

## Techniques Implemented

* Image reshaping and normalization (pixel values scaled to [0, 1])
* Data augmentation with `ImageDataGenerator` (rotation, zoom)
* CNN architecture: 3 convolutional layers + max pooling + dense layers
* FFNN architecture: flatten + 2 dense hidden layers
* SVM with a linear kernel on flattened image vectors
* Model evaluation: accuracy, classification report (precision/recall/F1), confusion matrix
* Visualization of confusion matrices (heatmaps) and per-model sample predictions

## Project Structure

```
mnist-classification/
├── notebooks/
│   └── mnist_classification.ipynb   # Full analysis notebook (Python)
└── README.md
```

## How to Use

1. Clone the repository and open the notebook in Jupyter or Google Colab:

```
jupyter notebook notebooks/mnist_classification.ipynb
```

2. Run the cells in order. TensorFlow will automatically download the MNIST dataset on first run — no manual data loading required.

3. Each model (CNN, FFNN, SVM) is trained and evaluated independently, with confusion matrices and sample predictions generated at the end.

## Author
Rawan Mansour

This was a team project.
