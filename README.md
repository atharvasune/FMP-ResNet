# ResNet + Fractional Max Pooling

## Introduction

This repository implements two research papers for state of the art Convolutional Neural Networks on the CIFAR - 10 dataset. Further, these two models have been combined to form a hybrid model. 

### ResNet

  * ResNet is a fairly recent architecture (2015) which has surpassed human level
performance on the ImageNet dataset.
  * ResNet solves the problem of vanishing gradients that occurs in many deep
CNNs, by implementing skip connections.
  * Typical skip connections are built over 2 or 3 layers. That is a skip connection is
present every 2 or 3 layers.


### Fractional Max Pooling

  * Traditionally neural nets use max pooling with 2x2 grids (2MP), which reduces the image dimensions by a factor of 2.
  * An alternative would be to use pooling schemes that reduce by factors other than two, e.g. `1 < factor < 2`.
  * Pooling by a factor of `sqrt(2)` would allow twice as many pooling layers as 2MP, resulting in "softer" image size reduction   throughout the network. This is called Fractional Max Pooling (FMP).
  * FMP has been shown to successfully remove overfitting without even using dropout.

## Papers

a. <a href = "https://arxiv.org/abs/1512.03385">Deep Residual Learning for Image Recognition</a>, by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

b. <a href = "https://arxiv.org/abs/1412.6071">Fractional Max-Pooling<a/> by Benjamin Graham

## The Hybrid Model
...

