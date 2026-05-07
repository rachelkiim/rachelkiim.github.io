---
layout: default
title: "04 Image Classification"
permalink: /vision/CS231N/04
#subtitle: 
use_math: true
parent: CS231N
grand_parent: vision
---

# 4. Image Classification


> [!summary] Summary
> image classification with linear classifier
> - algebraic viewpoint : $f(x, W) = Wx + b$
> - visual viewpoint : one template per class 
> - geometric viewpoint : hyperplanes cutting up space 
> to have a good $W$ : we use loss function and optimization


### Neural Networks 
the original linear classifier 

why do we want non-linearity? 
- in some cases: cannot separate data with linear classifier 
- after applying feature transform, points can be separated by linear classifier

Neural Networks also called as fully connected network or MLP 
* before linear score function : $F = Wx$
- now 2-layer Neural Network : $F = W_2max(0, W_1x)$
- now 3-layer Neural Network : $F = W_3max(0, W_2max(0, W_1x))$
	- function $max(0,z)$ = activation function 
	- if there's no activation function - we will end up with a **linear classification** again 

Activation functions: ReLU, Sigmoid, Tanh, GELU, Leaky ReLU ••• 
- problem of ReLU : non-positive에 대해 all-zero .. so Leaky ReLU comes out 
-  But ReLU is a good default choice for most problems 

#### 2-layer NN 
full implementation of training a 2-layer NN needs ~ 20 lines
``` python
import numpy as np
from numpy.random import randn

# define the network 
N, D_in, H, D_out = 64, 1000, 100, 10
x, y = randn(N, D_in), randn(N, D_out)
w1, w2 = randn(D_in, H), randn(H, D_out)

for t in range(2000):
	# forward pass 
	h = 1 / (1 + np.exp(-x.dot(w1)))
	y_pred = h.dot(w2)
	loss =  np.square(y_pred - y).sum()
	print(t, loss)

	# calcuate the analytical gradients 
	grad_y_pred = 2.0 * (y_pred - y)
	grad_w2 = h.T.dot(grad_y_pred)
	grad_h = grad_y_pred.dot(w2.T)
	grad_w1 = x.T.dot(grad_h * h * (1-h)) 
	
	# gradient descent 
	w1 -= 1e-4 * grad_w1
	w2 -= 1e-4 * grad_w2
```

https://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html demo 
- larger NN can represent more complicated function
- but do not use size of NN as a regularizer. 
- use stronger regularization instead ex. hyperparameter


### Neuron 
Biological Neurons
- complex connectivity patterns, and there are many difference types 
- dendrites can perform complex non-linear computations 
- synapses are not a single weight but a complex non-linear dynamical system 
neurons in a NN = organized into regular layers for computational efficiency 
- but NN with random connections can work too .. 


### loss function and NN 

#### Old idea 
**nonlinear score function** : $s = f(x; W_1, W_2) = W_2 \max(0, W_1 x)$ 
**hinge loss on predictions** : $L_i = \sum_{j \ne y_i} \max(0, s_j - s_{y_i} + 1)$ 
**regularization** : $R(W) = \sum_k W_k^2$ 
**Total Loss** : data loss +regularization
$L = \frac{1}{N} \sum_{i=1}^{N} L_i + \lambda R(W_1) + \lambda R(W_2)$
- if we can compute $\frac{\partial L}{\partial W_1}, \frac{\partial L}{\partial W_2}$ then we can learn $W_1,W_2$ 

problem
- very tedious : lots of matrix calculus, need lots of paper
- what if we want to change loss (softmax instead of hinge) .. then we need to re-derive everything from scratch 
- not feasible for very complex model

#### Better idea : computational graphs + backpropagation
