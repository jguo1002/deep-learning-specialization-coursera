# Coursera Deep Learning Specialization
Coursera deep learning specialization notes and code

[Coursera Specialization Page](https://www.coursera.org/specializations/deep-learning#courses)

Notes are taken in the format of QA. 

## Table of Content 

- [Course 1: Neural Networks and Deep Learning](#course-1-neural-networks-and-deep-learning)
    - [Week 3: Shallow Neural Networks](#week-3-shallow-neural-networks)
    - [Week 4: Deep Neural Networks](#week-4-deep-neural-networks)
- [Course 2: Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization](#course-2-improving-deep-neural-networks)
    - [Week 1: Practical Aspects of Deep Learning](#week-1-practical-aspects-of-deep-learning)
    - [Week 2: Optimization Algorithms](#week-2-optimization-algorithms)
    - [Week 3: Hyperparameter Tuning, Batch Normalization and Programming Frameworks]()



## Course 1: Neural Networks and Deep Learning 
[Coursera Syllabus](https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning#syllabus)

[YouTube Playlist](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0)

Week 1 is the overview of the course and specialization.

### Week 1 Introduction to Deep Learning

What is the differences between structure and unstructured data? 

|       | Features | Example |
| ----------- | ----------- | --- |
| Structured |  columns of database | house price
| Unstructured |  pixel value, individual word  | audio, image, text


### Week 2: Neural Networks Basics

What are the dimensions of input matrix and weights?

| Param      | Description | 
| ----------- | ----------- | 
| $m$ |  number of observations | 
| $n_x$ |  number of features (input data) |
| $L$ |  number of layers. $l=0$: input layer |
| $n^{[l]}$  | number of units (features) at layer $l$. $n^{[0]} = n_x$   |



| Matrix      | Shape | 
| ----------- | ----------- | 
| $X$ |      $(n_x, m)$ | 
| $W^{[l]}$   |   $(n^{[l]}, n^{[l-1]}) $  | 
| $Z^{[l]}$   |  $(n^{[l]}, m)$   |
| $A^{[l]}$   |  $(n^{[l]}, m)$   |



To better memberize

$W^{[l]}$:
```
num of row: number of units of the next layer
num of col: number of units of the current layer
```

$Z^{[l]}$ and $A^{[l]}$:
```
num of row: number of units of the next layer
num of col: number of observations
```


### Week 3: Shallow Neural Networks

Why do we use activation functions other than Sigmoid? What are the pros and cons of activation functions? 

| Activation Function      | Output | Characteristic |
| ----------- | ----------- | ----------- | 
| Sigmoid     | (0, 1)       |  good for output layer of binary classification   |
| Tanh        | (-1, 1)  |  center data, mean of data is close to 0, makes learning for next layer a bit easier  |
| ReLU        |  (0, $\infty$)   |  derivative of slope is 1 when z > 0,  is 0 when z < 0  | 
| Leasky ReLU  |  (-$\infty$, $\infty$)    |



Why non-linear activation functions? 
```
If we use linear activation functions, no matter how many layers you have, the NN is just computing a linear function. 
```


Why do we usually initialize W as small random values? 

```
large W -> large Z (Z = WX + b) -> end up at the flat parts of Sigmoid function 
-> gradient will be small -> gradient descent will be slow -> learning will be slow

If you're not using Sigmoid or Tanh activation functions, it is less of an issue. But note if you're doing a binary classification, the output layer will be a Sigmoid function. 
```

Logistic regression’s weights w should be initialized randomly rather than to all zeros, because if you initialize to all zeros, then logistic regression will fail to learn a useful decision boundary because it will fail to “break symmetry”, True/False?

```
Flase. Logistic Regression doesn't have a hidden layer. If you initialize the weights to zeros, the first example x fed in the logistic regression will output zero but the derivatives of the Logistic Regression depend on the input x (because there's no hidden layer) which is not zero. 

So at the second iteration, the weights values follow x's distribution and are different from each other if x is not a constant vector. 
```

>But in deep learning we should randomly initialize either W or b to "break symmetry". Using non-zero initialization but making them all the same does not work either. Though we can *learn* new values, but the values we get are symmetric, means it's the same as a network with a single neuron. 
>
>Reference: [Symmetry Breaking versus Zero Initialization](https://community.deeplearning.ai/t/symmetry-breaking-versus-zero-initialization/16061)


A = np.random.randn(4,3); B = np.sum(A, axis = 1, keepdims = True). 
What will be B.shape? 
```
(4, 1)
We use (keepdims = True) to make sure that A.shape is (4,1) and not (4, ). It makes our code more robust. 
```

### Week 4: Deep Neural Networks

What is the relationship between # of hidden units and # of layers? 

```
Informally: for equal performance shallower networks require exponentially more hidden units to compute. 
```

What is the intuition about deep representation? 

```
Intuitively, deeper layers compute more complex things such as eyes instead of edges. 
```

Vectorization allows you to compute forward propagation in an LL-layer neural network without an explicit for-loop (or any other explicit iterative loop) over the layers l=1, 2, …,L. True/False?
```
False. Forward propagation propagates the input through the layers, although for shallow networks we may just write all the lines. In a deeper network, we cannot avoid a for loop iterating over the layers.
```


## Course 2: Improving Deep Neural Networks


[Coursera Syllabus](https://www.coursera.org/learn/deep-neural-network?specialization=deep-learning#syllabus)

[YouTube Playlist](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc)

### Week 1: Practical Aspects of Deep Learning

#### `What are the differences when creating train, dev, test sets in traditional ML and DL?`

>In traditional ML, train/dev/test split may be 60% / 20% / 20%. 
>
>In DL, since the data is large, train/dev/test may be 99.5% / 0.4% / 0.1%
>
>Side note: not having a test set might be okay. 

`What should we do if the variance or bias is high? `

| Problem | Try |
| -- | -- | 
| High bias | Bigger network <br/> Train longer <br/> (NN architecture search)|
| High variance | More data <br/> Regularization <br/> (NN architecture search) | 


`Why regularization reduces overfitting? `
> If lambda is large, weights will be small or close to zero because gradient descent minimizes the cost function. 
>
> small weights -> decrease impacts of some hidden units -> simpler network -> not overfit


`What are the differences between L1 and L2 regularization? `

| Regularization | Penalize | $W$ | Feature selection | 
| -- | -- | -- | -- |
| L1 | sum of absolute values of the weights | sparse | Yes |
| L2 | sum of squares of the weights | non-sparse | No |  


`What is dropout regularization? Why does it work? `

> Dropout regularization randomly switch off some hidden units so they do not learn anything and the NN will be simpler  

`What should we pay attention to when implementing dropout during train / test time?`
|  | apply dropout | keep_prob | 
| -- | -- | -- |
| Train | Yes | Yes |
| Test | No | No | 
```
D1 = np.rand(a, b)
D1 = (D1 < keep_prob).astype(int)
A1 = A1 * D1 
A1 = A1 / keep_prob
```
> Note the devriatives during backward also need to scale 
```
dA1 = dA1 * D1
dA1 = dA1 / keep_prob
```


`What is weight decay?`
> A regularization technique (such as L2 regularization) that results in gradient descent shrinking the weights on every iteration.

`Why do we normalize the inputs x?`
> It makes the cost function easier and faster to optimize

`What is vanishing / exploding gradient ? `

> Derivatives of each layer are multiplied layer by layer (inputs times gradient). If we have a sigmoid or tanh activation function, derivates are always a fraction. 
> 
> During backpropagate, fractions are multiplying for many times, the gradient decreases expotentially and the weights of the initial layer will be very small, which makes it hard to learn. 

`How to deal with vanishing gradient?`

> A partial solution: force the variance of $W$ to be constant and smaller. A recommended value is $\frac{1}{n}$ but it depends on the activation function.  


### Week 2: Optimization Algorithms

What are the differences between batch , mini-batch, and stochatic gradient descent?


| GD | Size  | Train Time  |
| ------ | -------- | ----- |
| Batch  | m | too long |
| Stochatic | 1 | lose speed up by vectorization |
| Mini-batch | (1, m) | 



How to choose mini-batch size? 
```
if m <= 2000: 
    use batch gd
else: 
    typical size: 2_4, 2_5, 2_6... 

It depends on the context, we should test with different sizes 
```

`What is momentum?`
> Momentum is a method to dampen down the changes in gradients and accelerate gradients vectors in the right direction using exponentially weighted averages. 


### Week 3: Hyperparameter Tuning, Batch Normalization and Programming Frameworks



