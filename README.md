
# PyTorch Foundations Image & Text Classification

### Overview
This project serves as an introduction to PyTorch and Weights & Biases (wandb) by implementing and experimenting with deep learning models for image and text classification. The primary objectives include understanding PyTorch’s computation graphs, implementing a basic classifier, logging experiments with wandb, and modifying the baseline model to enhance performance.

## Project Structure
```
├── data/
│   ├── img_train.csv
│   ├── img_val.csv
│   ├── img_test.csv
│   ├── txt_train.csv
│   ├── txt_val.csv
│   ├── txt_test.csv
├── img_classifier.py
├── txt_classifier.py
├── README.md
```

## Getting Started

### Installation
To set up the environment locally, follow these steps:
1. Install Miniconda and create a Python environment:
   ```sh
   conda create -n py312 python=3.12
   conda activate py312
   ```
2. Install PyTorch (latest stable version):
   ```sh
   pip install torch torchvision torchaudio
   ```
3. Install Weights & Biases for experiment tracking:
   ```sh
   pip install wandb
   ```
4. Install additional dependencies:
   ```sh
   pip install pandas
   ```

For Google Colab, set the runtime type to GPU and mount Google Drive before running the code.

## Image Classification
This task involves classifying images into three categories: **Parrot, Narwhal, Axolotl**. The dataset consists of generated images, and the classifier is implemented using a simple feedforward neural network.

<img width="900" height="500" src="https://github.com/JavierAM01/PyTorch-Foundations-Image-Text-Classification/blob/main/images/montage.jpg">

### Model Implementation
The initial image classification model is a **fully connected feedforward neural network (MLP)** with the following architecture:
1. **Input Layer**: 256x256x3 flattened image vector.
2. **Hidden Layer 1**: Fully connected layer with 512 neurons and ReLU activation.
3. **Hidden Layer 2**: Fully connected layer with 512 neurons and ReLU activation.
4. **Output Layer**: Fully connected layer with 3 neurons (for classification) and a softmax function.

The loss function used is **CrossEntropyLoss**, and the optimizer used is **Stochastic Gradient Descent (SGD)**.

<img width="200" height="900" src="https://github.com/JavierAM01/PyTorch-Foundations-Image-Text-Classification/blob/main/images/new_model_graph.png">

### Experiments
- Implemented a **baseline classifier** with two hidden layers of 512 units each.
- Integrated Weights & Biases for logging.
- Modified the model to support grayscale images.
- Experimented with different optimizers (e.g., Adadelta vs. SGD).
- Implemented a convolutional neural network (CNN) to compare performance with the feedforward model.

<img width="500" height="250" src="https://github.com/JavierAM01/PyTorch-Foundations-Image-Text-Classification/blob/main/images/IMG_training_loss_SGD.png">

### Results Summary
| Model | Train Accuracy | Test Accuracy |
|--------|----------------|--------------|
| Baseline (MLP) | 98.22% | 73.19% |
| CNN Model | 95.78% | 74.22% |

## Text Classification
This task involves classifying news articles as **real or fake**, using an LSTM-based model and a simple bag-of-words model.

### Model Implementation
The baseline model consists of a **Bag-of-Words (BoW) representation** using `nn.EmbeddingBag`.
The improved model replaces this with an **LSTM-based classifier**, structured as follows:
1. **Embedding Layer**: Converts words to dense vectors.
2. **LSTM Layer**: Captures sequential dependencies in text.
3. **Pooling Layer**: Adaptive max pooling to aggregate features.
4. **Fully Connected Layer**: Maps the pooled LSTM outputs to class scores.

The loss function used is **CrossEntropyLoss**, and the optimizer used is **Adam**.

### Experiments
- Implemented a baseline classifier using `nn.EmbeddingBag`.
- Replaced it with an **LSTM-based classifier** with max-pooling.
- Switched from **SGD** to **Adam optimizer**.
- Analyzed performance differences and how padding affects LSTM models.

### Results Summary
| Model | Validation Accuracy | Test Accuracy |
|--------|-------------------|--------------|
| Baseline (Bag-of-Words) | ~92% | ~91% |
| LSTM Model | ~89% | ~88% |

## Training and Experimentation
Training was conducted over **multiple runs** with different hyperparameters and architectures. 
Each run was logged using **Weights & Biases (wandb)** to track training and validation performance.
- Loss curves and accuracy plots were analyzed to identify underfitting/overfitting trends.
- Hyperparameters such as learning rate, batch size, and optimizer choice were adjusted.
- Model architectures were iteratively refined to improve generalization.

## Key Learnings
- PyTorch’s `autograd` system enables easy computation of gradients.
- Weights & Biases helps in tracking experiment results effectively.
- CNNs generalize better than MLPs for image classification.
- LSTMs are sensitive to padding, and choosing the right architecture is crucial.

## How to Run the Code
1. Train the image classifier:
   ```sh
   python img_classifier.py
   ```
2. Train the text classifier:
   ```sh
   python txt_classifier.py
   ```
3. View experiment logs on Weights & Biases by navigating to [wandb.ai](https://wandb.ai/).

## Notes
- The dataset is synthetically generated for classification tasks.
- Some experiments, like grayscale image classification, show the impact of color information on predictions.

## Acknowledgments
This project is part of **10-623 Generative AI** at **Carnegie Mellon University**, with datasets and starter code provided by the course instructors.

