---
layout: default
title: "02 Regularization and Optimization"
permalink: /vision/CS231N/
#subtitle: 
use_math: true
parent: CS231N
grand_parent: vision
---

# 2. Regularization and Optimization

https://cs231n.github.io/optimization-1/
https://cs231n.github.io/optimization-2/


> [!summary] Summary 
> - we have some dataset of $(x, y)$
> - we have a score function : $s = f(x; W) = Wx$
> - we have a loss function (softmax, full loss, ••• )
> - finding the best W : optimize with Gradient Descent 


### Regularization 
$$
L(W) = \frac{1}{N} \sum_{i=1}^{N} L_i\big(f(x_i, W), y_i\big) + \lambda R(W)
$$
- 앞 항 : data loss (model predictions should match training data)
- 뒷 항 : regularization (prevent the model from doing too well on training data)
	- $\lambda$ : regularization strength (hyperparam)
	- **L2 regularization** : $R(W) = \sum_k \sum_l W_{k,l}^2$ (normally small value but nonzero)
	- **L1 regularization** : $R(W) = \sum_k \sum_l \lvert W_{k,l} \rvert$ (normally small value becomes zero)

#### Why regularize?
- express preferences over weights 
- make the model simple so it works on test data
- improve optimization by adding curvature 

#### examples 
$$
x = [1,1,1,1], w_1 = [1,0,0,0], w_2 = [0.25, 0.25, 0.25, 0.25] , w_1^Tx = w_2^Tx = 1 
$$
Q. which of w1 or w2 will the L2 regularizer prefer? 
A. loss same (dot product) but L2 will prefer w2 (nonzero)
Q. which one would L1 regularization prefer? 
A. same (both sum to 1). 


### Optimization
#### 1. Random search 
randomly search w .. very bad idea solution 
``` python 
bestloss = float("inf") # Python assigns the highest possible float value
for num in range(1000):
  W = np.random.randn(10, 3073) * 0.0001 # generate random parameters
  loss = L(X_train, Y_train, W) # get the loss over the entire training set
  if loss < bestloss: # keep track of the best solution
    bestloss = loss
    bestW = W
  print 'in attempt %d the loss was %f, best %f' % (num, loss, bestloss)
  
scores = Wbest.dot(Xte_cols) # 10 x 10000, the class scores for all test examples
Yte_predict = np.argmax(scores, axis = 0)
np.mean(Yte_predict == Yte)
```
- 15.5% accuracy (SOTA is ~ 99.7%)

#### 2. Follow the slope 
1-dimension 
$$ 
\frac{df(x)}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

multiple dimension : gradient is the vector of (partial derivative) along each dimension 
```python 
def eval_numerical_gradient(f, x):
  fx = f(x) # evaluate function value at original point
  grad = np.zeros(x.shape)
  h = 0.00001

  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    ix = it.multi_index
    old_value = x[ix]
    x[ix] = old_value + h # increment by h
    fxh = f(x) # evalute f(x + h)
    x[ix] = old_value # restore to previous value (very important!)

    # compute the partial derivative
    grad[ix] = (fxh - fx) / h # the slope
    it.iternext() # step to next dimension

  return grad
```

- numerical gradient : approximate, slow, easy to write 
- analytic gradient : exact, fast, error-prone
- in practice: always use analytic gradient, but check implementation with numerical gradient (gradient check)

### Gradient Descent 

gradient of the loss function = procedure of repeatedly evaluating the gradient and then performing a parameter update 
#### 1. vanilla gradient descent 
``` python 
while True:
  weights_grad = evaluate_gradient(loss_fun, data, weights)
  weights += - step_size * weights_grad # perform parameter update
```
- how do we know where to stop? 

#### 2. mini-batch gradient descent
```python 
while True:
  data_batch = sample_training_data(data, 256) # sample 256 examples
  weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
  weights += - step_size * weights_grad # perform parameter update
```

#### 3. Stochastic gradient descent
$$  
L(W) = \frac{1}{N} \sum_{i=1}^{N} L_i(x_i, y_i, W) + \lambda R(W)  
$$
- full sum is **expensive** when N is large  
$$  
\nabla_W L(W) =  
\frac{1}{N} \sum_{i=1}^{N} \nabla_W L_i(x_i, y_i, W)  
+ \lambda \nabla_W R(W)  
$$
- approximate sum using a **minibatch** of examples 

#### problem with SGD
1. What if loss changes **quickly in one direction and slowly in another**? 
	- step size가 너무 커서 value를 벗어날 수 있음
	- steep direction으로 가지 않아서 slow progress가 될 수 있음 
2. what if the loss function has a **local minim**a or saddle point? 
	- saddle points much more common in high dimension 
3. our gradients come from minibatches so they **can be noisy** 
→ SGD + Momentum 

#### SGD + Momentum 
continue moving to the general direction as the previous iterations 
SGD : $x_{t+1} = x_t - \alpha \nabla f(x_t)$
```python
while True:
	dx = compute_gradient(x)
	x -= learning_rate * dx
```

SGD + Momentum : $v_{t+1} = \rho v_t + \nabla f(x_t), x_{t+1} = x_t - \alpha v_{t+1}$
```python
vx = 0
while True:
	dx = compute_gradient(x)
	vx = rho * vs + dx
	x -= learning_rate * vx
```

#### more complex optimizers: RMSProp 
adds element-wise scaling of the gradient based on the historical sum of squares in each dimension 
```python
grad_squared = 0
while True:
	dx = compute_gradient(x)
	grad_squared = decay_rate * grad_square + (1 - decay_rate) * dx * dx
	x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
```

#### Adam 
sort of like RMSProp with momentum 
bias correction for the fact that first and second moment estimate start at zero 
```python 
first_moment = 0 
second_moment = 0
while True:
	dx = compute_gradient(x)
	first_moment = beta1 * first_moment + (1-beta1) * dx # momentum 
	second_moment = beta2 * second_moment + (1-beta2) * dx * dx # RMSProp
	first_unbias = first_moment / (1-beta1 ** t) # bias correction 
	second_unbias = second_moment / (1-beta2 ** t)
	x -= learning_rate * first_memoent / (np.sqrt(second_moment) + 1e-7) 
```

SGD, SGD+Momentum, RMSProp, Adam, AdamW all have learning rate as a hyperparameter
Q. which one of these learning rates is best to use?
A. In reality, all of these could be good learning rates 

#### learning rate decay 
reduce learning rate at a few fixed point (used in ResNet training)
- multiply learning rate by 0.1 after epochs 30, 60, 90
- cosine learning rate decay 

