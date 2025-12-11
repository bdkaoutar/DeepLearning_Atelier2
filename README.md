# Atelier 2 – Part 1  
## Deep Learning Architectures on MNIST (CNN, Faster R-CNN, VGG16, AlexNet)

This notebook explores several deep learning architectures applied to the MNIST dataset, ranging from a classical Convolutional Neural Network to advanced transfer-learning models. The objective is to understand how different architectures perform on a simple classification task and how their design impacts accuracy, robustness, and training time.

---

## **Models Implemented**

### **1. Custom CNN (Baseline Model)**
- Built from scratch using PyTorch
- Uses convolution, pooling, batch normalization, and dropout
- Trained for digit classification on grayscale MNIST images

### **2. Faster R-CNN (Object Detection)**
- Originally designed for detection tasks  
- Adapted here to detect and classify digits with auto-generated bounding boxes  
- Demonstrates why detection models are not optimal for pure classification workloads

### **3. VGG16 (Transfer Learning)**
- ImageNet-pretrained VGG16 adapted to MNIST  
- MNIST images converted to RGB and resized to 224×224  
- Fully fine-tuned to evaluate the benefit of deep pretrained features

### **4. AlexNet (Transfer Learning)**
- Similar transfer-learning setup as VGG16  
- Faster to train, slightly lighter architecture

---

## **Metrics Used for Comparison**
Each model was compared using:
- **Accuracy**
- **F1-score**
- **Training time**
- **Loss evolution**

These metrics help evaluate precision, robustness, and efficiency across different neural network families.

---

## **Key Takeaways**

- The **custom CNN** performs very well for MNIST with fast training and high accuracy, making it the best efficiency choice.
- **Faster R-CNN** works but performs poorly compared to classifiers. Its heavy detection pipeline makes it slow and unnecessary for single-digit classification.
- **VGG16** achieves the **highest overall accuracy** thanks to transfer learning, but requires significant computational time.
- **AlexNet** performs well but slightly below VGG16.
- Transfer learning provides strong benefits even on simple datasets.

---

## **What I Learned**
During this lab, I learned:
- How to build and train convolutional networks from scratch
- How image detection frameworks (Faster R-CNN) differ from pure classifiers
- How to prepare data for pretrained models (RGB conversion, resizing)
- How transfer learning boosts performance and when it is appropriate to use
- How to compare models using quantitative metrics to make informed decisions

This notebook provides a full practical overview of model training pipelines, evaluation, and performance trade-offs in deep learning.

---

# Atelier 2 – Part 2  
## Vision Transformer (ViT) Implementation on MNIST

This notebook introduces a from-scratch implementation of a **Vision Transformer (ViT)** applied to the MNIST handwritten digits dataset. The goal is to understand how transformer-based architectures can be adapted for image classification tasks and how they compare conceptually to convolutional networks.

---

## Overview of the Architecture

The notebook builds the entire ViT model manually using PyTorch, including:

### **1. Patch Embedding**
- The input image (28×28) is divided into **patches of size 7×7**.  
- Each patch is flattened and projected to a vector of size 128 using a convolutional projection layer.  
- This produces a sequence of patches treated similarly to tokens in NLP models.

### **2. Class Token + Positional Encoding**
- A learnable **CLS token** is prepended to the patch sequence.  
- Learnable **positional embeddings** are added to preserve spatial information.

### **3. Transformer Encoder**
A stack of encoder blocks, each containing:
- **Layer Normalization**
- **Multi-Head Self-Attention**
- **MLP feed-forward network**
- **Residual connections**

These blocks allow the model to capture global dependencies across image patches.

### **4. Classification Head**
- The output corresponding to the **CLS token** is passed through a linear layer.  
- The network outputs a distribution over the **10 MNIST classes**.

---

## Dataset
The MNIST dataset is loaded directly from IDX files and preprocessed into:
- Grayscale images shaped as **(1, 28, 28)**
- Integer labels from **0 to 9**

Two DataLoaders are used:
- **train_loader** (batch size 128)  
- **test_loader** (batch size 128)

---

## Evaluation Metrics
After training the Vision Transformer, the following metrics were computed:

- **Accuracy**
- **F1-score (macro)**

These metrics assess classification performance and balance across all classes.

---

## Results

The manually implemented ViT achieves:

- **Competitive accuracy on MNIST**  
- **Strong F1-score** given the lightweight architecture and small image size  

This illustrates the effectiveness of transformer-based models even on simple image datasets.

---

## What I Learned

During this lab, I learned:

- How Vision Transformers process images through patch embedding instead of convolution
- How self-attention enables global reasoning across image regions
- How to implement multi-head attention, normalization, and residual blocks manually
- How to construct a ViT encoder and classification pipeline end-to-end
- How transformer architectures differ conceptually from CNNs, and when each approach is more appropriate

This notebook provides a hands-on understanding of transformer mechanics and their application to computer vision tasks.

---

## Summary

This second part of the atelier deepens the understanding of modern neural architectures by building a Vision Transformer from scratch. Through this exercise, I gained insight into the power, flexibility, and design patterns behind transformer models used in state-of-the-art computer vision systems.
