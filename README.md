# Basic machine learning models for reference

Algorithms benchmarked to the MNIST dataset

## Principle Component Analysis (PCA)

Notes: 

## Artificial Neural Network (ANN)

Simple fully-connected network

MNIST accuracy: **96%**

## Convolutional Neural Network (CNN)

Simple convolutional neural network

MNIST accuracy: **98%**

Notes: Slow, only ran for 2 epoch. 

## Convolutional Neural Network (VGG16)

Implementation of the VGG16 architecture with pretrained MNIST data

![VGG16](./media/vgg16.png)
<small> cite: https://www.cs.toronto.edu/~frossard/post/vgg16/ </small>

*Note:* Because MNIST is 28x28, first 10 layers omitted and layer depth made a factor of 3 smaller (as MNIST has only 1 color layer)

MNIST accuracy: 

## Stacked Autoencoder (X-wing AE)

A symmetric autoencoder for unsupervised learning 

Notes: Success is largely dependent on layer sizes, seems to have a small window of convergence

## Convolutional Autoencoder (CAE)

A symmetric convolutional autoencoder

## Variational Autoencoder (VAE)
