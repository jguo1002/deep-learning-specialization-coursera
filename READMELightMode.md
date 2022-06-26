# Coursera Deep Learning Specialization
Coursera deep learning specialization notes and code

[Coursera Specialization Page](https://www.coursera.org/specializations/deep-learning#courses)

Notes are taken in the format of QA. 

## Table of Contents

- [Course 1: Neural Networks and Deep Learning](#course-1-neural-networks-and-deep-learning)
    - [Week 3: Shallow Neural Networks](#week-3-shallow-neural-networks)
    - [Week 4: Deep Neural Networks](#week-4-deep-neural-networks)
- [Course 2: Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization](#course-2-improving-deep-neural-networks)
    - [Week 1: Practical Aspects of Deep Learning](#week-1-practical-aspects-of-deep-learning)
    - [Week 2: Optimization Algorithms](#week-2-optimization-algorithms)
    - [Week 3: Hyperparameter Tuning, Batch Normalization and Programming Frameworks](#week-3-hyperparameter-tuning-batch-normalization-and-programming-frameworks)
- [Course 3: Structuring Machine Learning Projects](#course-3-structuring-machine-learning-projects)
- [Course 4: Convolutional Neural Networks](#course-4-convolutional-neural-networks)
    - [Week 1: Foundations of Convolutional Neural Networks](#week-1-foundations-of-convolutional-neural-networks)
    - [Week 2: Deep Convolutional Models: Case Studies](#week-2-deep-convolutional-models-case-studies)
    - [Week 3: Detection Algorithms](#week-3-detection-algorithms)
    - [Week 4: Face Recognition](#week-4-face-recognition)
- [Course 5: Sequence Model](#course-5-sequence-model)
    - [Week 1: Recurrent Neural Network](#week-1-recurrent-neural-network)
    - [Week 2: Natural Language Processing & Word Embeddings](#week-2-natural-language-processing--word-embeddings)




## Course 1: Neural Networks and Deep Learning 
[Coursera Syllabus](https://www.coursera.org/learn/neural-networks-deep-learning?specialization=deep-learning#syllabus)

[YouTube Playlist](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0)

Week 1 is the overview of the course and specialization.

### Week 1 Introduction to Deep Learning

**What is the difference between structure and unstructured data?** 

|       | Features | Example |
| ----------- | ----------- | --- |
| Structured |  columns of database | house price
| Unstructured |  pixel value, individual word  | audio, image, text


### Week 2: Neural Networks Basics

**What are the dimensions of input matrix and weights?**

| Param      | Description | 
| ----------- | ----------- | 
| <img src="https://render.githubusercontent.com/render/math?math=m"> |  number of observations | 
| <img src="https://render.githubusercontent.com/render/math?math=n_x"> |  number of features (input data) |
| <img src="https://render.githubusercontent.com/render/math?math=L"> |  number of layers. <img src="https://render.githubusercontent.com/render/math?math=l=0">: input layer |
| <img src="https://render.githubusercontent.com/render/math?math=n^{[l]}">  | number of units (features) at layer <img src="https://render.githubusercontent.com/render/math?math=l">. <img src="https://render.githubusercontent.com/render/math?math=n^{[0]} = n_x">   |



| Matrix      | Shape | 
| ----------- | ----------- | 
| <img src="https://render.githubusercontent.com/render/math?math=X"> |      <img src="https://render.githubusercontent.com/render/math?math=(n_x, m)"> | 
| <img src="https://render.githubusercontent.com/render/math?math=W^{[l]}">   |   <img src="https://render.githubusercontent.com/render/math?math=(n^{[l]}, n^{[l-1]}) ">  | 
| <img src="https://render.githubusercontent.com/render/math?math=Z^{[l]}">   |  <img src="https://render.githubusercontent.com/render/math?math=(n^{[l]}, m)">   |
| <img src="https://render.githubusercontent.com/render/math?math=A^{[l]}">   |  <img src="https://render.githubusercontent.com/render/math?math=(n^{[l]}, m)">   |



To better memberize

<img src="https://render.githubusercontent.com/render/math?math=W^{[l]}">:
```
num of row: number of units of the next layer
num of col: number of units of the current layer
```

<img src="https://render.githubusercontent.com/render/math?math=Z^{[l]}"> and <img src="https://render.githubusercontent.com/render/math?math=A^{[l]}">:
```
num of row: number of units of the next layer
num of col: number of observations
```
[⬆️ Back to top](#table-of-contents)


### Week 3: Shallow Neural Networks

**What are the pros and cons of activation functions?**

| Activation Function      | Output | Characteristic |
| ----------- | ----------- | ----------- | 
| Sigmoid     | (0, 1)       |  good for output layer of binary classification   |
| Tanh        | (-1, 1)  |  center data, mean of data is close to 0, makes learning for next layer a bit easier  |
| ReLU        |  (0, <img src="https://render.githubusercontent.com/render/math?math=\infty">)   |  derivative of slope is 1 when z > 0,  is 0 when z < 0  | 
| Leasky ReLU  |  (-<img src="https://render.githubusercontent.com/render/math?math=\infty">, <img src="https://render.githubusercontent.com/render/math?math=\infty">)    |



**Why non-linear activation functions?** 

> If we use linear activation functions, no matter how many layers you have, the NN is just computing a linear function. 


**Why do we usually initialize <img src="https://render.githubusercontent.com/render/math?math=W"> as small random values?** 

> large W -> large Z (Z = WX + b) -> end up at the flat parts of Sigmoid function 
-> gradient will be small -> gradient descent will be slow -> learning will be slow
>
>If you're not using Sigmoid or Tanh activation functions, it is less of an issue. But note if you're doing a binary classification, the output layer will be a Sigmoid function. 

**What distribution should we draw <img src="https://render.githubusercontent.com/render/math?math=W"> from?**

> Normal distirbution. 
>
> In Python we should use `np.random.randn` (normal distribution) instead of `np.random.rand` (uniform distribution). 

**Logistic regression’s weights w should be initialized randomly rather than to all zeros, because if you initialize to all zeros, then logistic regression will fail to learn a useful decision boundary because it will fail to “break symmetry”, True/False?**


> False. Logistic Regression doesn't have a hidden layer. If you initialize the weights to zeros, the first example x fed in the logistic regression will output zero but the derivatives of the Logistic Regression depend on the input x (because there's no hidden layer) which is not zero. 
> 
> So at the second iteration, the weights values follow x's distribution and are different from each other if x is not a constant vector. 
>
> But in deep learning we should randomly initialize either <img src="https://render.githubusercontent.com/render/math?math=W"> or <img src="https://render.githubusercontent.com/render/math?math=b"> to "break symmetry". 
If both <img src="https://render.githubusercontent.com/render/math?math=W"> and <img src="https://render.githubusercontent.com/render/math?math=b"> values zero, <img src="https://render.githubusercontent.com/render/math?math=A^{[1]}"> will be 0 because *tanh(0)=0*. 
>Using non-zero initialization but making them all the same does not work either. Though we can *learn* new values, but the values we get are symmetric, means it's the same as a network with a single neuron. 
>
>Reference: [Symmetry Breaking versus Zero Initialization](https://community.deeplearning.ai/t/symmetry-breaking-versus-zero-initialization/16061)


**A = np.random.randn(4,3); B = np.sum(A, axis = 1, keepdims = True). 
What will be B.shape?**

<details>
    <summary>Click to see answer</summary>
    
> (4, 1)
> 
>We use (keepdims = True) to make sure that A.shape is (4,1) and not (4, ). It makes our code more robust. 

</details>


[⬆️ Back to top](#table-of-contents)

### Week 4: Deep Neural Networks

**What is the relationship between # of hidden units and # of layers?** 

> Informally: for equal performance shallower networks require exponentially more hidden units to compute. 

**What is the intuition about deep representation?**

> Intuitively, deeper layers compute more complex things such as eyes instead of edges. 

**Vectorization allows you to compute forward propagation in an LL-layer neural network without an explicit for-loop (or any other explicit iterative loop) over the layers l=1, 2, …,L. True/False?**



> False. Forward propagation propagates the input through the layers, although for shallow networks we may just write all the lines. In a deeper network, we cannot avoid a for loop iterating over the layers.


[⬆️ Back to top](#table-of-contents)

## Course 2: Improving Deep Neural Networks


[Coursera Syllabus](https://www.coursera.org/learn/deep-neural-network?specialization=deep-learning#syllabus)

[YouTube Playlist](https://www.youtube.com/playlist?list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc)

### Week 1: Practical Aspects of Deep Learning

**What are the differences when creating train, dev, test sets in traditional ML and DL?**

>In traditional ML, train/dev/test split may be 60% / 20% / 20%. 
>
>In DL, since the data is large, train/dev/test may be 99.5% / 0.4% / 0.1%
>
>Side note: not having a test set might be okay. 

**What should we do if the variance or bias is high?**

| Problem | Try |
| -- | -- | 
| High bias | Bigger network <br/> Train longer <br/> (NN architecture search)|
| High variance | More data <br/> Regularization <br/> (NN architecture search) | 


**Why regularization reduces overfitting?**
> If lambda is large, weights will be small or close to zero because gradient descent minimizes the cost function. 
>
> small weights -> decrease impacts of some hidden units -> simpler network -> not overfit


**What are the differences between L1 and L2 regularization?**

| Regularization | Penalize | <img src="https://render.githubusercontent.com/render/math?math=W"> | Feature selection | 
| -- | -- | -- | -- |
| L1 | sum of absolute values of the weights | sparse | Yes |
| L2 | sum of squares of the weights | non-sparse | No |  


**What is dropout regularization? Why does it work?**

> Dropout regularization randomly switch off some hidden units so they do not learn anything and the NN will be simpler  

**What should we pay attention to when implementing dropout during train / test time?**
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


**What is weight decay?**
> A regularization technique (such as L2 regularization) that results in gradient descent shrinking the weights on every iteration.

**Why do we normalize the inputs x?**
> It makes the cost function easier and faster to optimize

**What is vanishing / exploding gradient ?**

> Derivatives of each layer are multiplied layer by layer (inputs times gradient). If we have a sigmoid or tanh activation function, derivates are always a fraction. 
> 
> During backpropagate, fractions are multiplying for many times, the gradient decreases expotentially and the weights of the initial layer will be very small, which makes it hard to learn. 

**How to deal with vanishing gradient?**

> A partial solution: force the variance of <img src="https://render.githubusercontent.com/render/math?math=W"> to be constant and smaller. A recommended value is <img src="https://render.githubusercontent.com/render/math?math=\frac{1}{n}"> but it depends on the activation function.  

[⬆️ Back to top](#table-of-contents)

### Week 2: Optimization Algorithms

What are the differences between batch , mini-batch, and stochatic gradient descent?


| GD | Size  | Train Time  |
| ------ | -------- | ----- |
| Batch  | m | too long |
| Stochatic | 1 | lose speed up by vectorization |
| Mini-batch | (1, m) | 

**How to choose mini-batch size?**
```
if m <= 2000: 
    use batch gd
else: 
    typical size: 2_4, 2_5, 2_6... 

It depends on the context, we should test with different sizes 
```

**Formula of bias correction in exponentially weighted averages?**

<details>
    <summary>Click to see answer</summary>

><img src="https://render.githubusercontent.com/render/math?math=v_t = \beta v_{t-1} + (1-\beta) \theta_t">
>
> <img src="https://render.githubusercontent.com/render/math?math=v^{corrected}_t = \frac{v_t}{1-\beta^t}">

</details>

<br />

**What is momentum?**

> Momentum is a method to dampen down the changes in gradients and accelerate gradients vectors in the right direction using exponentially weighted averages. 

**Which of these is NOT a good learning rate decay scheme? Here, t is the epoch number.**

1. <img src="https://render.githubusercontent.com/render/math?math=\alpha=0.95^t\alpha_0">
2. <img src="https://render.githubusercontent.com/render/math?math=\alpha=e^t\alpha_0">
3. <img src="https://render.githubusercontent.com/render/math?math=\alpha=\frac{1}{1+2*t}\alpha_0">
4. <img src="https://render.githubusercontent.com/render/math?math=\alpha=\frac{1}{\sqrt{t}}\alpha_0">

<details>
    <summary>Click to see answer</summary>

> <img src="https://render.githubusercontent.com/render/math?math=\alpha=e^t\alpha_0"> explodes <img src="https://render.githubusercontent.com/render/math?math=\alpha"> instead of decaying it

</details>

<br />

**What is the process of parameter update in Adam?**

<details>
    <summary>Click to see answer</summary>

> <img src="https://render.githubusercontent.com/render/math?math=v_{dW^{[l]}} = \beta_1 v_{dW^{[l]}} + (1-\beta_1) \frac{\partial J}{\partial W^{[l]}}">
> 
> <img src="https://render.githubusercontent.com/render/math?math=v_{dW^{[l]}} ^{correted} = \frac{v_{dW^{[l]}}}{1 - (\beta_1)^t}">
>
> <img src="https://render.githubusercontent.com/render/math?math=s_{dW^{[l]}} = \beta_2 v_{dW^{[l]}} + (1-\beta_2) (\frac{\partial J}{\partial W^{[l]}}) ^2">
>
> <img src="https://render.githubusercontent.com/render/math?math=s_{dW^{[l]}} ^{correted} = \frac{s_{dW^{[l]}}}{1 - (\beta_2)^t}">
> 
> <img src="https://render.githubusercontent.com/render/math?math=W^{[l]} = W^{[l]} - \alpha \frac{v^{corrected}_{dW^{[l]}}}  {\sqrt{s_{dW^{[l]}} ^{correted}} + \varepsilon}">

</details>

<br />

### Week 3: Hyperparameter Tuning, Batch Normalization and Programming Frameworks


**Why batch normalization?**

> Normalization can make training faster and hyperparameters more robust. 
>
> Values of each hidden layer are changing all the time because of changes in <img src="https://render.githubusercontent.com/render/math?math=W"> and <img src="https://render.githubusercontent.com/render/math?math=b">, suffering from the problem of covariance shift. Batch normalization guarantees the mean and variance of features of each layer (e.g., <img src="https://render.githubusercontent.com/render/math?math=Z^{[2]}_1">, <img src="https://render.githubusercontent.com/render/math?math=Z^{[2]}_2">) keep the same no matter how actual values of each node changes. 
> It allows each layer to learn by itself (more independently than no batch normalization), and speed up learning. 
>
> Mean and variance are governed by two learnable parameters <img src="https://render.githubusercontent.com/render/math?math=\gamma"> and <img src="https://render.githubusercontent.com/render/math?math=\beta">. Adding <img src="https://render.githubusercontent.com/render/math?math=\gamma"> and <img src="https://render.githubusercontent.com/render/math?math=\beta"> is because we don't want all the layers have the same mean and variance (mean = 0, variance = 1).


**Batch normalization fomula?**

<details>
    <summary>Click to see answer</summary>

> <img src="https://render.githubusercontent.com/render/math?math=z_{norm}^{(i)} = \frac{z^{(i) - \mu}}{\sqrt{\sigma^2 + \epsilon}}">

</details>

<br />


**If searching among a large number of hyperparameters, should you try values in a gird or by random? Why**

> Random. 
>
> Grid method is okay if # of hyperparameter is small. 
> In DL, it is difficult to know in advance which hyperparameter is more important. Random method allow us to try more distinct values that are potentially important. 

**What the hyperparamers and their default values?**

| Hyperparameter | common value | 
| --- | --- |
| learning rate <img src="https://render.githubusercontent.com/render/math?math=\alpha"> | <img src="https://render.githubusercontent.com/render/math?math=r\in [-4,0], \alpha=10^r"> | 
| momentum <img src="https://render.githubusercontent.com/render/math?math=\beta"> | around 0.9 | 
| mini-batch size | <img src="https://render.githubusercontent.com/render/math?math=2^n"> | 
| # of hidden units | - |
| learning rate decay | <img src="https://render.githubusercontent.com/render/math?math=10^r"> |
| # of layers <img src="https://render.githubusercontent.com/render/math?math=L"> | - | 
| batch normalization <img src="https://render.githubusercontent.com/render/math?math=\beta_1">, <img src="https://render.githubusercontent.com/render/math?math=\beta_2">, <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> | 0.9, 0.99, <img src="https://render.githubusercontent.com/render/math?math=10^{-8}"> | 


[⬆️ Back to top](#table-of-contents)


## Course 3: Structuring Machine Learning Projects

### Week 1

**What are the types of metrics?**

> Optimizing metric: the metric you want as good as possible, e.g., accuracy
> 
> Satisficing metric: as long as it reaches a threshold, e.g., run time, memory

**How should we make decisions on train/dev set error ?**

> We should always have Bayes error to estimate avoidable bias. Human-level error is often seen as a proxy of Bayes eror.
> 
> A learning algorithm’s performance can be better human-level performance but it can never be better than Bayes error. human-level performance

**We should not add data from a different distribution to the `train` set. True / False?**

> False.
> 
> Sometimes we'll need to train the model on the data that is available, and its distribution may not be the same as the data that will occur in production. Also, adding training data that differs from the dev set may still help the model improve performance on the dev set. What matters is that the dev and test set have the same distribution.

**We should not add data from a different distribution to the `test` set. True / False?**

> True. 
> 
> This would cause the dev and test set distributions to become different.

**What should you do if another metric (e.g., false negative rate) should be taken into account?**

> Rethink the appropriate metric for this task, and ask your team to tune to the new metric.

### Week 2

**A softmax activation would be a good choice for the output layer if this is a multi-task learning problem. True/False?**

> False.
> Softmax would be a good choice if one and only one of the possibilities (stop sign, speed bump, pedestrian crossing, green light and red light) was present in each image.

**Should you correct mislabeled data in train and test set after you did so in dev set?**

> You should correct mislabeled data in test set because test and dev set should come from the same distribution. 
> 
> You do not necessarily need to fix the mislabeled data in the train set because it's okay for the train set distribution to differ from the dev and test sets. 


**Let's say you have 100,000 images taken by cars' front camera (you care about) and 900,000 images from the internet. How should you split train/test set?**

> One example: 
> 
> Train set: 900,000 images from the internet + 80,000 images from car’s front-facing camera. 
> 
> Dev / Test set: The 20,000 remaining front-camera images in each set.
> 
> As seen in lecture, it is important that your dev and test set have the closest possible distribution to “real”-data. It is also important for the training set to contain enough “real”-data to avoid having a data-mismatch problem.


[⬆️ Back to top](#table-of-contents)

<br>

## Course 4: Convolutional Neural Networks

### Week 1: Foundations of Convolutional Neural Networks


**What are the problems of convolution?**

> 1. Each time you apply a convolution operator, the image shrinks. 
> 2. Pixels on the corner or edge will be used much less than those in the middle. 

**Notations and dimensions of input matrix and parameters**

| Param      | Description | 
| ----------- | ----------- | 
| <img src="https://render.githubusercontent.com/render/math?math=f^{[l]}"> |  filter size |
| <img src="https://render.githubusercontent.com/render/math?math=p^{[l]}"> | padding |
| <img src="https://render.githubusercontent.com/render/math?math=s^{[l]}"> | stride | 
| <img src="https://render.githubusercontent.com/render/math?math=n_c^{[l]}"> | number of filters | 

| Metric      | Dimension | 
| ----------- | ----------- | 
| Filter |  <img src="https://render.githubusercontent.com/render/math?math=(f^{[l]}, f^{[l]}, n_c^{[l]})"> |
| Activations | <img src="https://render.githubusercontent.com/render/math?math=(n_H^{[l]}, n_W^{[l]}, n_c^{[l]})"> |
| Weights | <img src="https://render.githubusercontent.com/render/math?math=(f^{[l]}, f^{[l]}, n_c^{[l-1]}, n_c^{[l]})"> | 
| bias | <img src="https://render.githubusercontent.com/render/math?math=(1, 1, 1, n_c^{[l]})"> | 

**What is valid and same convolutions?**

> Valid: no padding
> 
> Same: Pad so that output size is the same as the input size

**How to calculate the dimension of next conv layer?**

> <img src="https://render.githubusercontent.com/render/math?math=\lfloor \frac{n+2p-f}{s} +1 \rfloor \times \lfloor \frac{n+2p-f}{s} +1 \rfloor">

**Input is a 300 by 300 color (RGB) image, and you use a convolutional layer with 100 filters that are each 5x5. How many parameters does this hidden layer have (including the bias parameters)?**

> (5 * 5 * 3 + 1) * 100 = 7,600
> 
> Each filter is a volume where the number of channels matches up the number of channels of the input volume.

**What are the benefits of CNN?**
> 1. It allows a feature detector to be used in multiple locations throughout the whole input volume.
> 
> 2. Convolutional layers provide sparsity of connections.

**What does “sparsity of connections” mean?**
> Each activation in the next layer depends on only a small number of activations from the previous layer.
> 
> Yes, each activation of the output volume is computed by multiplying the parameters from only one filter with a volumic slice of the input volume and then summing all these together. 



[⬆️ Back to top](#table-of-contents)


### Week 2: Deep Convolutional Models: Case Studies

**LeNet - 5**

> LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). [Gradient-based learning applied to document recognition](http://web.khu.ac.kr/~tskim/NE%2009-2%20LeNet%201998.pdf). Proceedings of the IEEE, 86(11), 2278-2324.

*hard to read*

**AlexNet**

> Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). [Imagenet classification with deep convolutional neural networks](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html). Advances in neural information processing systems, 25.

*easy to read*

**VGG - 16**

> Simonyan, K., & Zisserman, A. (2014). [Very deep convolutional networks for large-scale image recognition](https://arxiv.org/pdf/1409.1556.pdf). arXiv preprint arXiv:1409.1556.

**ResNet**

> He, K., Zhang, X., Ren, S., & Sun, J. (2016). [Deep residual learning for image recognition](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf). In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

**What does skip-connection do?**

> Skip-connections make it easy for the network to learn an identity mapping between the input and the output within the ResNet block. 

**Why does ResNet work?**

> It helps with gradient vanishing and exploding problems and allows people to train deep neural networks without loss in performance. 

> "The skip connections in ResNet solve the problem of vanishing gradient in deep neural networks by allowing this alternate shortcut path for the gradient to flow through. The other way that these connections help is by allowing the model to learn the identity functions which ensures that the higher layer will perform at least as good as the lower layer, and not worse. "
> 
> -- [Introduction to Resnet or Residual Network](https://www.mygreatlearning.com/blog/resnet)


**Inceptionm Network**

> Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A. (2015). [Going deeper with convolutions](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html). In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1-9).


**MobileNet**

> Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., ... & Adam, H. (2017). [Mobilenets: Efficient convolutional neural networks for mobile vision applications](https://arxiv.org/pdf/1704.04861.pdf). arXiv preprint arXiv:1704.04861.



**Suppose that in a MobileNet v2 Bottleneck block we have an <img src="https://render.githubusercontent.com/render/math?math=n\times n \times 5"> input volume. We use 30 filters for the expansion. In the depthwise convolutions we use <img src="https://render.githubusercontent.com/render/math?math=3 \times 3"> filters, and 20 filters for the projection.**

**How many parameters are used in the complete block, suppose we don't use bias?**

> Expansion filter: 5 * 30 = 150
> 
> Depthwise: 3 * 3 * 30 = 270 
> 
> Pointwise: 30 * 20 = 600
> 
> Total = 150 + 270 + 600 = 1020




**EfficientNet**

> Tan, M., & Le, Q. (2019, May). [Efficientnet: Rethinking model scaling for convolutional neural networks](http://proceedings.mlr.press/v97/tan19a/tan19a.pdf). In International conference on machine learning (pp. 6105-6114). PMLR.

[⬆️ Back to top](#table-of-contents)

### Week 3: Detection Algorithms

**YOLO**

> Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). [You only look once: Unified, real-time object detection](https://arxiv.org/pdf/1506.02640.pdf). In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 779-788).
>
> Github: [YAD2K: Yet Another Darknet 2 Keras](https://github.com/allanzelener/YAD2K)


**What are advantages of YOLO algorithm?**

> - It can ouput more accurate bounding box
> - It's one single convolutional computation where you use one conv net with lots of shared computation between all the computation needed for all the cells. Therefore it's a very efficient algorithm. 

**How to evaluate object localization?**

> <img src="https://render.githubusercontent.com/render/math?math=IoU = \frac{size\ of\ intersection}{size\ of\ union} ">

**How does non-max suppression work?**

> While BoundingBoxes: 
>  
> - Pick the box with the highest score (<img src="https://render.githubusercontent.com/render/math?math=p_c * c_i">), add it to ouput list 
> - Calculate the overlap IoU of this box and all the other boxes. Discard boxes that overlap significantly (`iou >= iou_threshold`). 
> - Repeat the steps above until there are no more boxes with a lower score than the currently selected box.

**What is the dimension of one grid in YOLO? Suppose there are <img src="https://render.githubusercontent.com/render/math?math=C"> classes and <img src="https://render.githubusercontent.com/render/math?math=A"> anchors**

> <img src="https://render.githubusercontent.com/render/math?math=A \times (C + 5)"> 
> 
> <img src="https://render.githubusercontent.com/render/math?math=5: [p_c, b_x, b_y, b_h, b_w]">

**How does Transposed Convolution work?**

Transposed Convolutions are used to upsample the input feature map to a desired output feature map using some learnable parameters.

```
- pick the top left corner element of input, multiply it with every element in the kernel
- put the result (the same size with kernel) on the top left corner of output matrix
- pick the second element of input, multiple it with every element in the kernel
- put the result in the output matrix based on stride
- repeat the steps above 
- if there is overlap of results, add the elements 
- ignore elements in the padding
```
Read more: 

[Make your own neural network | Calculating the Output Size of Convolutions and Transpose Convolutions](http://makeyourownneuralnetwork.blogspot.com/2020/02/calculating-output-size-of-convolutions.html)

[Towards Data Science | Transposed Convolution Demystified](https://towardsdatascience.com/transposed-convolution-demystified-84ca81b4baba)

**U-net**
> Ronneberger, O., Fischer, P., & Brox, T. (2015, October). [U-net: Convolutional networks for biomedical image segmentation](https://arxiv.org/pdf/1505.04597.pdf). In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.

**What is the dimension of U-Net archiecture output?**

> <img src="https://render.githubusercontent.com/render/math?math=h \times w \times k"> where <img src="https://render.githubusercontent.com/render/math?math=k"> is the number of classes

[⬆️ Back to top](#table-of-contents)

### Week 4: Face Recognition


**What is the differences between face verification and face recognition?**

| Input | Output | Comparison |
| -- | -- | -- |
| An image and a name/ID | whether the input image if the claimed person | 1:1 |
| An image | if the image is any of the K persons | 1:K |

**Siamese Network**

> Taigman, Y., Yang, M., Ranzato, M. A., & Wolf, L. (2014). [Deepface: Closing the gap to human-level performance in face verification](https://openaccess.thecvf.com/content_cvpr_2014/papers/Taigman_DeepFace_Closing_the_2014_CVPR_paper.pdf). In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1701-1708).

**Triplet Loss**
> Schroff, F., Kalenichenko, D., & Philbin, J. (2015). [Facenet: A unified embedding for face recognition and clustering](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf). In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 815-823).

**Neural Style Transfer**

> Zeiler, M. D., & Fergus, R. (2014, September). [Visualizing and understanding convolutional networks](https://link.springer.com/content/pdf/10.1007/978-3-319-10590-1_53.pdf). In European conference on computer vision (pp. 818-833). Springer, Cham.

> Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). [A neural algorithm of artistic style](https://arxiv.org/abs/1508.06576). arXiv preprint arXiv:1508.06576.

**What is the cost function of style transfer?**

> <img src="https://render.githubusercontent.com/render/math?math=J(G) = \alpha J_{content}(C, G) + \beta J_{style}(S, G)">
>
> <img src="https://render.githubusercontent.com/render/math?math=G">: generated image.
>  
> <img src="https://render.githubusercontent.com/render/math?math=C">: content image. 
> 
> <img src="https://render.githubusercontent.com/render/math?math=S">: style image. 
> 
> <img src="https://render.githubusercontent.com/render/math?math=G_{gram}">: gram matrix
> 
> Content cost. For each layer: 
>
> <img src="https://render.githubusercontent.com/render/math?math=J_{content}(C,G)^{[l]} = \frac{1}{2} \Vert a^{[l](C)} - a^{[l](G)} \Vert^2 ">
> 
> Content cost. For all the entries
>
> <img src="https://render.githubusercontent.com/render/math?math=J_{content}(C,G) = \frac{1}{4 \times n_H \times n_W \times n_C} \sum_{all entries}( a^{(C)} - a^{(G)} )^2 ">
> 
> Style cost. For each layer: 
>
> <img src="https://render.githubusercontent.com/render/math?math=J^{[l]}_{style}(S, G) = \frac{1}{(2 n^{[l]}_H n^{[l]}_W n^{[l]}_C)^2}    \sum\limits_{i=1}^{n_C}     \sum\limits_{j=1}^{n_C}     (G^{[l](S)}_{(gram) i, j} - G^{[l](G)}_{(gram) i,j})^2 ">
> 
> Style cost. For all entries: 
> 
> <img src="https://render.githubusercontent.com/render/math?math=J_{style} (S,G) = \sum\limits_l \lambda^{[l]} J ^{[l]}_{style} (S,G) ">

[⬆️ Back to top](#table-of-contents)

## Course 5: Sequence Model

### Week 1: Recurrent Neural Network

**Why not use standard network on sequence data?**

> - Inputs and outputs can be different in length. 
> - Standard network doesn't share features across positions of text. E.g., Harry at position 0 is a name, is other Harry at other positions also a name?

**Notations**

| Param      | Description | 
| ----------- | ----------- | 
| <img src="https://render.githubusercontent.com/render/math?math=x^{(i)<t>}"> |  the t th element in the training sequence i |
| <img src="https://render.githubusercontent.com/render/math?math=T_x^{(i)}"> | the length of training sequence i |
| <img src="https://render.githubusercontent.com/render/math?math=y^{(i)<t>}"> | the t th element in the output sequence i | 
| <img src="https://render.githubusercontent.com/render/math?math=T_y^{(i)}"> | the length of output sequence i |

**What is the formula of forward propagation?**

> <img src="https://render.githubusercontent.com/render/math?math=a^{<t>} = g(W_{aa}a^{<t-1>} + W_{ax}x^{<t>} + b_a)">
>
> <img src="https://render.githubusercontent.com/render/math?math=y^{<t>} = g(W_{ya}a^{<t>} + b_y)">
> 
> Here <img src="https://render.githubusercontent.com/render/math?math=W_{ax}">, the second index means <img src="https://render.githubusercontent.com/render/math?math=W_{ax}"> will be multiplied by some x-like quantity, to compute some a-like quantity. 

**List some examples of RNN architectures**

> - Many to one: sentiment classiciation 
> - One to many: music generation. Input: genre / first note; output: a sequence of notes
> - Many to many (different length): machine translation. 

[Andrej Karpathy: The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)


**Why RNN has vanishing gradients problems?**

> An ouput is mainly influenced by values close to its position.  
> It's difficult for an output to be strongly influenced by an input that is very early in the sequence. Because it's difficult to backpropagate all the wway to the beginning of the sequence. 

**How to deal with exploding gradients?**
> Apply gradients clipping. Re-scale some gradient vectors when it's bigger than some threshold. 

**Where and how do you apply clipping?**

> forward pass -> cost computation -> backward pass -> CLIPPING -> parameter update
> 
> `np.clip(gradient, -maxValue, maxValue, out = gradient)`
> 
```
def optimize(X, Y, a_prev, parameters, learning_rate): 

    loss, cache = rnn_forward(X, Y, a_prev, parameters)
    
    gradients, a = rnn_backward(X, Y, parameters, cache)
    
    gradients = clip(gradients, 5)
    
    parameters = update_parameters(parameters, gradients, learning_rate)
        
    return loss, gradients, a[len(X)-1]

```


**What is the formula of Gated Recurrent Unit (GRU)?**

> <img src="https://render.githubusercontent.com/render/math?math=\tilde{c}^{<t>} = tanh(W_c[\Gamma_r * c^{<t-1>}, x^{<t>}] + b_c)">
>
> <img src="https://render.githubusercontent.com/render/math?math=\Gamma_u">: updated gate, (0,1)
>
> <img src="https://render.githubusercontent.com/render/math?math=\Gamma_u = \sigma(W_u[c^{<t-1>}, x^{<t>}] + b_u)"> 
> 
> <img src="https://render.githubusercontent.com/render/math?math=\Gamma_r">: how relevant <img src="https://render.githubusercontent.com/render/math?math=c^{<t-1>}"> is to <img src="https://render.githubusercontent.com/render/math?math=\tilde{c}^{<t>}">. Same update method with <img src="https://render.githubusercontent.com/render/math?math=\Gamma_u">
>
> 
> <img src="https://render.githubusercontent.com/render/math?math=c^{<t>} = \Gamma_u * \tilde{c}^{<t>} + (1-\Gamma_u) * c^{<t-1>}">

Cho, K., Van Merriënboer, B., Bahdanau, D., & Bengio, Y. (2014). [On the properties of neural machine translation: Encoder-decoder approaches.](https://arxiv.org/pdf/1409.1259.pdf)

Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). [Empirical evaluation of gated recurrent neural networks on sequence modeling.](https://arxiv.org/pdf/1412.3555.pdf)


Hochreiter, S., & Schmidhuber, J. (1997). [Long short-term memory. Neural computation, 9(8), 1735-1780.](https://www.researchgate.net/profile/Sepp-Hochreiter/publication/13853244_Long_Short-term_Memory/links/5700e75608aea6b7746a0624/Long-Short-term-Memory.pdf)

**How does LSTM differ from GRU?**

> Instead of having one update gate controls <img src="https://render.githubusercontent.com/render/math?math=\tilde{c}^{<t>}"> and <img src="https://render.githubusercontent.com/render/math?math=c^{<t-1>}">, LSTM has two separate gates <img src="https://render.githubusercontent.com/render/math?math=\Gamma_u"> and <img src="https://render.githubusercontent.com/render/math?math=\Gamma_f"> (forget gate). 
>
> Update <img src="https://render.githubusercontent.com/render/math?math=c^{<t>}">: 
>
> <img src="https://render.githubusercontent.com/render/math?math=c^{<t>} = \Gamma_u * \tilde{c}^{<t>} + \Gamma_f * c^{<t-1>}">

**What are the disadvantages of Bidirectional RNN?**

> You do need an entire sequence of date before making predictions anywhere (cannot use in real-time application).

**True/False: In RNN, step t uses the probabilities output by the RNN to pick the highest probability word for that time-step. Then it passes the ground-truth word from the training set to the next time-step.**

> No, the probabilities output by the RNN are not used to pick the highest probability word and the ground-truth word from the training set is not the input to the next time-step.

**You find your weights and activations are all taking on the value of NaN (“Not a Number”), what problem may cause it?**

> Gradient exploding. It happens when large error gradients accumulate and result in very large updates to the NN model weights during training. These weights can become too large and cause an overflow, identified as NaN.

**Sarah proposes to simplify the GRU by always removing the <img src="https://render.githubusercontent.com/render/math?math=\Gamma_u">. I.e., setting <img src="https://render.githubusercontent.com/render/math?math=\Gamma_u"> = 0. Ashely proposes to simplify the GRU by removing the <img src="https://render.githubusercontent.com/render/math?math=\Gamma_r">. I. e., setting <img src="https://render.githubusercontent.com/render/math?math=\Gamma_r">= 1 always. Which of these models is more likely to work without vanishing gradient problems even when trained on very long input sequences?**

> No. For the signal to backpropagate without vanishing, we need c<t> to be highly dependent on <img src="https://render.githubusercontent.com/render/math?math=c^{<t−1>}">.

[⬆️ Back to top](#table-of-contents)


### Week 2: Natural Language Processing & Word Embeddings

**What is the downside of skip-gram model?**

> The Softmax objective is expensive to compute because it needs to sum over the entire vocabulary. 

**What are differences of problem objectives in the skip-gram model and negative sampling?**

> Skip-gram: given a context, predict the probability of different target word 
> 
> Negative sampling: given a pair of words, is it a context-target pair? Is it a positive or negative sample? 

**Why negative sampling's computation cost is lower?**

> It converts a N softmax problem to a N binary classification problem. 
> In each iteration, only train K words. K = 5 to 20 in small vocabulary, K = 2 to 5 in large vobabulary. 


**What is the learning objective of GloVe?**

> <img src="https://render.githubusercontent.com/render/math?math=minimize \sum{}^{10,000}_{i=1} \sum{}^{10,000}_{j=1} f(X_{ij}) (\theta_j^T e_j + b_i + b'_j - logX_{ij} )^2 ">
>
> <img src="https://render.githubusercontent.com/render/math?math=X_{ij}"> = # of times <img src="https://render.githubusercontent.com/render/math?math=j"> appears in the context of <img src="https://render.githubusercontent.com/render/math?math=i">
> 
> depending on the definition of "context", <img src="https://render.githubusercontent.com/render/math?math=X_{ij}"> and <img src="https://render.githubusercontent.com/render/math?math=X_{ji}"> may be symmetric. 

**Debiasing word embeddings**

Bolukbasi, T., Chang, K. W., Zou, J. Y., Saligrama, V., & Kalai, A. T. (2016). [Man is to computer programmer as woman is to homemaker? debiasing word embeddings](). Advances in neural information processing systems, 29.


**What are the steps in the debiasing word embeddings paper?**

> - Step 1: Identify gender subspace by SVD.
> - Step 2a: Hard de-biasing (neutralize and equalize).
> - Step 2b: Soft bias correction.
> 
> Determine gender specific words: first listed218 words from dictionary, then trained a SVM to classify 3M words in w2vNEWS, resulting in 6,449 gender-specific words. 

**<img src="https://render.githubusercontent.com/render/math?math=A"> is an embedding matrix, <img src="https://render.githubusercontent.com/render/math?math=o_{4567}"> is a one-hot vector corresponding to word 4567. Can we call <img src="https://render.githubusercontent.com/render/math?math=A * o_{4567}"> in Python to get the embedding of word 4567?**

> The element-wise multiplication is extremely inefficient. 


**What the four steps of sampling?**

> 1. Input the "dummy" vector of zeros  <img src="https://render.githubusercontent.com/render/math?math=𝑥^{<1>}=\vec{0}"> and <img src="https://render.githubusercontent.com/render/math?math=a^{<0>}=\vec{0}">
> 2. Run one step of forward pass to get <img src="https://render.githubusercontent.com/render/math?math=a^{<t+1>}"> and <img src="https://render.githubusercontent.com/render/math?math=\hat{y}^{<t+1>}">
> 3. Sampling the next index with the probability in <img src="https://render.githubusercontent.com/render/math?math=\hat{y}^{<t+1>}">. Use `np.random.choice`
> 4. Update to <img src="https://render.githubusercontent.com/render/math?math=x^{<t>}">. Set `x[idx] = 1`