---
layout: default
title: "01 Image Classification with linear classifiers"
permalink: /vision/CS231N/
#subtitle: 
use_math: true
parent: CS231N
grand_parent: vision
---

# 1. Image Classification with linear classifiers

https://cs231n.github.io/classification/
https://cs231n.github.io/linear-classify/


> [!summary] Summary 
> **Challenges of recognition** : viewpoint, illumination, deformation, occlusion, clutter, intraclass variation 
> **Various Approach** : data-driven, kNN 
> **Distance Metric** : L1 distance, L2 distance 
> **Linear Classifier** : $f(x, W)=Wx+b$

### The problem: Semantic Gap 
- our perception <-→ how the machine percepts 
#### Major challenges 
- **if the camera moves** : all the pixel values will be changed (have new value even the picture is same) 
- **illumination** : same objects may seen difference way in difference conditions 
- **occlusion** : human knows by context , but machine don't. 
- **deformation** : cats are really deformative ..
- **intraclass variation** : cats can come in difference sizes or colors .. many difference types of objects, each with their own appearance 
- **context** : e.g. 그림자. background, ••• 

#### Attempts 
- find edges, find corners ,, limitation : finding logics for requirements are so challenging 
- ML should be data-driven approaches 

#### ML: Data-Driven Approach 
1. collect a dataset of images and labels 
2. use ML algorithms to train a classifier
3. evaluate the classifier on new images 


### Nearest Neighbor Classifier
```python
# Memorize all data and labels 
def train(images, labels):
	# ML 
	return model

# Predict the label of the most similar training images 
def predict(model, test_images):
	# Use model to predict labels
	return test_labels
```

training data with labels -- **<u>distance metric</u>** -→ R 
- e.g. L1 distance : test image - training image → pixel-wise absolute value differences

```python
import numpy as np

class NearestNeighbor:
	def __init__(self):
		pass
	
	# Memorize training data 
	def train(self, X, y):
		self.Xtr = X
		self.ytr = y
	
	# for each test image ..
	# find closet train image, predict label of nearest image 
	def predict(self, X):
		num_test = X.shape[0]
		Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
		
		for i in xrange(num_test):
	      # find the nearest training image to the i'th test image
	      # using the L1 distance (sum of absolute value differences)
			distance = np.sum(np.abs(self.Xtr = X[i, :]), axis=1)
			min_index = np.argmin(distances)
			Ypred[i] = self.ytr[min_index]
		
		return Ypred
```

Q. With N examples, how fast are training and prediction? 
A. Train O(1), predict O(N) 
- not really good ,, 
- we want classifier that are fast at prediction, slow for training is ok 

### K-Nearest Neighbor Classifier 
find the top-k closest images, and have them vote on the label of the test image 
- k = 1 : same as Nearest Neighbor classifier
- **higher values of k** have a smoothing effect that makes the classifier more resistant to outliers

#### Distance Metric
- same distance from zero-point to the line (square or circle)
http://vision.stanford.edu/teaching/cs231n-demos/knn/ demo for knn 

### Hyperparameters 
hyperparameters
- what's the best value of k to use? 
- what's the best distance to use?
- often some of variable that I should make a decision of 
- very problem/dataset-dependent. must try them all out and see what works best 

#### Setting Hyperparameters 
- IDEA 1 : choose hyperparams that work best on the **training data**
  → BAD : K=1 cus always works perfectly on training data 
- IDEA 2 : choose hyperparams that work best on **test data**
  → BAD : kinda cheeting .. don't know how the model works for other data that are not in the test data (generalization problem)
- IDEA 3 : split data into train / val - **choose hyperparams on val and evaluate on test** 
- IDEA 4 : cross-validation : split data into folds, try each fold as val and average the results 


### Line Classifier
#### parametric approach 
image (array of 32x32x3=3072 numbers) → f(x, W) → 10 numbers giving class scores 
- W : parameters or weight 
- $f(x, W)$ (10x1) = $W$ (10x3072) · $x$ (3072x1) $+  b$ (10x1) 

#### find a good W 
1. define a loss function that quantifies our unhappiness with the scores across the training data
	- loss function = how good our current classifier is. 
2. come up with a way of efficiently finding the params that minimize the loss function (optimization)

#### softmax classifier
multinomial logistic regression
want to interpret raw classifier scores as probabilities $s = f(x_i; W)$
- numbers → exp → normalize .. then get the numbers that sum to 1 (probabilities)
- then compare probabilities with correct probs (1, 0, 0, ••• )
	- minimize KL-divergence $D_KL(P||Q)$ , cross Entropy $H(P,Q)$