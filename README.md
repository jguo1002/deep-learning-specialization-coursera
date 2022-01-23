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
    - [Week 2: Optimization Algorithms]()
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
|   |   |

| Matrix      | Shape | 
| ----------- | ----------- | 
| $X$ |      $(n_x, m)$ | 
| $W^{[l]}$   |   $(n^{[l]}, n^{[l-1]}) $  | 
| $Z^{[l]}$   |  $(n^{[l]}, m)$   |
| $A^{[l]}$   |  $(n^{[l]}, m)$   |
|   |   |

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
| Leasky ReLU  |  (-$\infty$, $\infty$)    |         |



Why non-linear activation functions? 
```
If we use linear activation functions, no matter how many layers you have, the NN is just computing a linear function. 
```


Why do we usually initialize W as small random values? 

```
large W -> large Z (Z = WX + b) -> end up at the flat parts of Sigmoid function -> gradient will be small -> gradient descent will be slow -> learning will be slow

If you're not using Sigmoid or Tanh activation functions, it is less of an issue. But note if you're doing a binary classification, the output layer will be a Sigmoid function. 
```

Logistic regression’s weights w should be initialized randomly rather than to all zeros, because if you initialize to all zeros, then logistic regression will fail to learn a useful decision boundary because it will fail to “break symmetry”, True/False?

```
Flase.
Logistic Regression doesn't have a hidden layer. If you initialize the weights to zeros, the first example x fed in the logistic regression will output zero but the derivatives of the Logistic Regression depend on the input x (because there's no hidden layer) which is not zero. So at the second iteration, the weights values follow x's distribution and are different from each other if x is not a constant vector. 
```

A = np.random.randn(4,3); B = np.sum(A, axis = 1, keepdims = True). 
What will be B.shape? 
```
(4, 1)
We use (keepdims = True) to make sure that A.shape is (4,1) and not (4, ). It makes our code more robust. 
```

### Week 4: Deep Neural Networks




## Course 2: Improving Deep Neural Networks
### Week 1: Practical Aspects of Deep Learning

[Coursera Syllabus](https://www.coursera.org/learn/deep-neural-network?specialization=deep-learning#syllabus)

[YouTube Playlist](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc)




### Week 2: Optimization Algorithms

### Week 3: Hyperparameter Tuning, Batch Normalization and Programming Frameworks



