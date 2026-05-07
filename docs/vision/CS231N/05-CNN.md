---
layout: default
title: "05 CNN Architecture"
permalink: /vision/CS231N/
#subtitle: 
use_math: true
parent: CS231N
grand_parent: vision
---

# 5. CNN Architecture

> [!summary] Summary Training Non-Recurrent NN
> 
> 1. one time setup : activation functions, preprocessing, weight initialization, normalization, transfer learning
> 2. training dynamics: babysitting the learning process, parameter updates, hyperparameter optimization
> 3. evaluation : validation performance, test-time augmentation

how to build CNNs? : layers in CNNs, Activation Functions, CNN architectures, weight initialization how to train CNNs? : data preprocessing, data augmentation, transfer learning, hyperparameter selection

## how to build CNNs

### Dropout

add randomization during the training process

- goal : make model harder to learn the data (better generalization)
- force the network to have a redundant representation
- prevents co-adaptation of features
- probability of dropping is a hyperparameter (0.5 is common)

test time

```python
def predict(x):
	# ensembled forward pass
	H1 = np.maximum(0, np.dot(W1, X) + b1) * p
	H2 = np.maximum(0, np.dot(W2, H1) + b2) * p
	out = np.dot(W3, H2) + b3
```

at test time - all neurons are active always

- we must scale the activations so that for each neuron → output at test time = expected output at training time
- train때 일부 뉴런을 끄기 때문에 test 때는 전체를 쓰되 값을 줄여서 평균을 맞춰줘야 한다는 것

### Activation Functions

$$

sigmoid : \sigma(x) = \frac{1}{1 + e^{-x}}

$$

- squashes numbers to range [0, 1]
- nice interpretation as a saturating 'firing rate' of a neuron
- problem: large positive / negative values can kill the gradients ..

$$ ReLU : f(x) = max(0, x) $$

- does not saturate in + region
- very computationally efficient
- converges much faster than sigmoid
- problem: not zero-centered output & Dead ReLUs when $x<0$

### VGGNet

- smaller filters, deeper networks
- 16-19 layers (AlexNet has 8 layers)
- only 3x3 Conv stride 1, pad 1 and 2x2 Max Pool stride 2
- What is the ERF of three 3x3 conv (stride 1) layers?
    - A3 (1x1) <- A2 (3x3) <- A1 (5x5) <- input (7x7) 
	

### ResNet

- deep models have more representation power than shallower models (they have more params)
- hypothesis: the problem is an optimization problem (deep models are hard to optimize)
- solution
    - use network layers to fit a residual mapping instead of directly trying to fit a desired underlying mapping.
    - deep NN이 무너지는 것을 방지하기 위해 원래 입력을 유지하면서 필요한 변화만 더함.
    - identify mapping $H(x) = x$ if $F(x) = 0$
- Full Architecture
    - stack residual blocks
    - every residual blocks - two 3x3 conv layers
    - double # of filters has stride 2

### Weight Initialization

```python
dims = [4096] * 7
hs = []
x = np.random.rand(16, dims[0])
# forward pass with ReLU activation
for Din, Dout in zip(dims[:-1], dims[1:]):
	W = 0.01 * np.random.rand(Din, Dout) # small weight init
	x = np.maximum(0, x.dot(W)) # ReLU activation
	hs.append(x)
```

- if we change 0.01 to 0.05 ,, activations blow up quickly
- how can we know the right value for training? → depends on the size of the layer `W = np.random.randn(Din, Dout * np.sqrt(2/Din)`

## How to train CNNs

### Data preprocessing

TLDR for image normalization

- subtract per-channel mean
- divide by per-channel std
- requires pre-computing means and std for each pixel channel

### Data augmentation

load image and label → cat → compute loss

- horizonal flips, random crops and scales

### Transfer learning

if we don't have a lot of data .. → 이미 배운 모델을 가져와서 새로운 문제에 재사용하자

1. ImageNet으로 pretraining
2. Small Dataset (C classes)로 training
3. Bigger Dataset으로 training

### choosing hyperparameters

1. check initial loss
2. overfit a small sample
3. find LR that makes loss go down
4. coarse grid of hyperparmas, train for ~ 1 - 5 epochs
5. refine grid, train longer
6. look at loss and acc. curves