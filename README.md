# ResNet + Fractional Max Pooling

## Introduction

This repository implements two research papers for state-of-the-art Convolutional Neural Networks on the CIFAR - 10 dataset. Further, these two models have been combined to form a hybrid model. The performances of all these datasets have been documented and shown below.

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

  * <a href = "https://arxiv.org/abs/1512.03385">Deep Residual Learning for Image Recognition</a>, by Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

  * <a href = "https://arxiv.org/abs/1412.6071">Fractional Max-Pooling</a> by Benjamin Graham

## Models

<h4> ResNet </h4>

  * Number of Layers (ResNet blocks): 20
  * Number of Dense Layers: 2
  * Number of Parameters: 1,612,042
  * train accuracy: ~78%
  * validation accuracy: ~71%
  * <a href = "https://github.com/AtharvaSune/FMP-ResNet/blob/master/ResNEt/ResNet.txt">Model Summary </a>
  
<h4> Fractional Max Pooling </h4>

  * Number of Layers (Convolutional blocks): 18
  * Number of Dense Layers: 2
  * Number of Parameters: 66,865,738
  * train accuracy: ~70%
  * validation accuracy: ~69%
  * <a href = "https://github.com/AtharvaSune/FMP-ResNet/blob/master/FMP/FMP.txt">Model Summary </a>
  
<h4> HYBRID: ResNet + Fractional Max Pooling </h4>

This model aims to combine both the above techniques. To accomplish this, the Max Pooling layers in ResNet have been replaced by Fractional Pooling Layers, which were created using Keras Layer Sub-classing (Custom Layer)
  * Number of Layers (ResNet blocks): 12
  * Number of Dense Layers: 2
  * Number of Parameters: 868,362
  * train accuracy: ~78%
  * validation accuracy: ~72%
  * <a href = "https://github.com/AtharvaSune/FMP-ResNet/blob/master/Hybrid/Hybrid.txt">Model Summary </a>

## Results

These models were evaluated on the CIFAR - 10 dataset with a training set size of 50,000 and a validation set size of 10,000.
Training was done over 50 epochs with a step size 100. There were 64 images per batch.
<br />

#### ResNet
![alt text](https://github.com/AtharvaSune/FMP-ResNet/blob/master/ResNEt/accuracy.png "ResNet")
![alt text](https://github.com/AtharvaSune/FMP-ResNet/blob/master/ResNEt/Loss.png "ResNet")
#### Fractional Max Pooling
![alt text](https://github.com/AtharvaSune/FMP-ResNet/blob/master/FMP/accuracy.png "Fractional Max Pooling")
![alt text](https://github.com/AtharvaSune/FMP-ResNet/blob/master/FMP/loss.png "Fractional Max Pooling")
#### Hybrid Model
![alt text](https://github.com/AtharvaSune/FMP-ResNet/blob/master/Hybrid/accuracy.png "Hybrid")
![alt text](https://github.com/AtharvaSune/FMP-ResNet/blob/master/Hybrid/loss.png "Hybrid")

## Conclusions

  * The Hybrid Model gives comparable results to the ResNet model, with almost <b>half</b> the number of layers and parameters.
  * This leads to much lesser scope of overfitting in the Hybrid Model since the number of parameters is 50 %.
  * Since the size of the model is significantly smaller, the memory consumption of the model is lesser but gives similar levels of performance.
  * For more complex models, the depth of the hybrid model can thus be increased without worrying about computation and overfitting as with regular ResNets

