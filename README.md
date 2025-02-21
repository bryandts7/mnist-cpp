
# Handwritten Digit Recognition using Neural Networks

This project implements a neural network model to recognize handwritten digits (0-9) using the MNIST dataset. The project is written in C++ and utilizes OpenCV for data processing and model training.

---

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Code Implementation](#code-implementation)
- [Reproduce the Project](#reproduce-the-project)
- [Results](#results)
- [Conclusions](#conclusions)
- [Usage](#usage)
- [Dependencies](#dependencies)

---

## Introduction
Handwritten digit recognition is a critical task in modern applications, such as processing bank checks and postal addresses. This project trains Artificial Neural Network (ANN) models to classify handwritten digits from the MNIST dataset. The neural network consists of interconnected nodes (neurons) with weights adjusted through forward propagation and backpropagation to minimize the loss function.

---

## Dataset
The MNIST dataset contains 60,000 grayscale images of handwritten digits (0-9), each of size 28x28 pixels. The dataset is divided into 10 classes, with pixel values ranging from 0 to 255.

- **Source**: [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- **Format**: IDX file format (images and labels are stored in `.idx3-ubyte` and `.idx1-ubyte` files, respectively).

---

## Code Implementation
The project is divided into two main modules:

### 1. **Data Extraction and Preprocessing**
   - **Files**: `mnist_utils.cpp`, `mnist_utils.h`
   - **Functions**:
     - `readIDX3UByteFile`: Reads image data from IDX3-UBYTE files.
     - `readLabelFile`: Reads label data from IDX1-UBYTE files.
     - `readMNISTFiles`: Extracts and processes MNIST dataset into OpenCV matrices and integer labels.

### 2. **Deep Learning Algorithm**
   - **Files**: `dnn_utils.cpp`, `dnn_utils.h`
   - **Functions**:
     - `trainingModel`: Constructs and trains a Deep Neural Network (DNN) model using OpenCV's `cv::ml::ANN_MLP`.
     - `testModelAccuracy`: Evaluates the accuracy of the trained model on test data.

---

## Reproduce the Project
To reproduce the project:
1. **Modify `main.cpp`**:
   - Uncomment lines 22-24 to train a new model.
   - Comment line 28 to avoid loading pre-trained models.
   - Customize hyperparameters (e.g., number of hidden layers, units) in `dnn_utils.cpp`.

2. **Use Pre-trained Models**:
   - Load pre-trained models (`trained_mnist_model_2HU_100.xml`, `trained_mnist_model_800HU.xml`) by modifying line 28 in `main.cpp`.

3. **Build and Run**:
   - Compile the project with OpenCV libraries.
   - Run the generated `main.exe` to see training and test accuracy.

---

## Results
The following results were obtained using different neural network architectures:

| Model                     | Training Accuracy | Test Accuracy |
|---------------------------|-------------------|---------------|
| 1 Hidden Layer (100 units) | 98.42%            | 96.78%        |
| 1 Hidden Layer (800 units) | 99.72%            | 97.08%        |
| 2 Hidden Layers (100+100)  | 99.33%            | 97.25%        |

---

## Conclusions
- **Hidden Layer Size**: Increasing hidden layer size improves training accuracy but may lead to overfitting.
- **Multiple Hidden Layers**: Two hidden layers provide better generalization compared to a single larger layer.
- **Trade-off**: Balancing model complexity and generalization is crucial for optimal performance.

---

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   ```
2. Navigate to the project directory:
   ```bash
   cd your-repo-name
   ```
3. Compile the project with OpenCV:
   ```bash
   g++ main.cpp mnist_utils.cpp dnn_utils.cpp -o main.exe `pkg-config --cflags --libs opencv4`
   ```
4. Run the executable:
   ```bash
   ./main.exe
   ```

---

## Dependencies
- **OpenCV**: Required for image processing and neural network implementation.
- **C++ Compiler**: Ensure your compiler supports C++11 or later.

---

## Acknowledgments
- MNIST dataset by Yann LeCun.
- OpenCV for providing the machine learning module.

---

For any questions or issues, feel free to open an issue or contact the author.
