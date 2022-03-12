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
| <img src="https://render.githubusercontent.com/render/math?math=\color{white}m"> |  number of observations | 
| <img src="https://render.githubusercontent.com/render/math?math=\color{white}n_x"> |  number of features (input data) |
| <img src="https://render.githubusercontent.com/render/math?math=\color{white}L"> |  number of layers. <img src="https://render.githubusercontent.com/render/math?math=\color{white}l=0">: input layer |
| <img src="https://render.githubusercontent.com/render/math?math=\color{white}n^{[l]}">  | number of units (features) at layer <img src="https://render.githubusercontent.com/render/math?math=\color{white}l">. <img src="https://render.githubusercontent.com/render/math?math=\color{white}n^{[0]} = n_x">   |



| Matrix      | Shape | 
| ----------- | ----------- | 
| <img src="https://render.githubusercontent.com/render/math?math=\color{white}X"> |      <img src="https://render.githubusercontent.com/render/math?math=\color{white}(n_x, m)"> | 
| <img src="https://render.githubusercontent.com/render/math?math=\color{white}W^{[l]}">   |   <img src="https://render.githubusercontent.com/render/math?math=\color{white}(n^{[l]}, n^{[l-1]}) ">  | 
| <img src="https://render.githubusercontent.com/render/math?math=\color{white}Z^{[l]}">   |  <img src="https://render.githubusercontent.com/render/math?math=\color{white}(n^{[l]}, m)">   |
| <img src="https://render.githubusercontent.com/render/math?math=\color{white}A^{[l]}">   |  <img src="https://render.githubusercontent.com/render/math?math=\color{white}(n^{[l]}, m)">   |



To better memberize

<img src="https://render.githubusercontent.com/render/math?math=\color{white}W^{[l]}">:
```
num of row: number of units of the next layer
num of col: number of units of the current layer
```

<img src="https://render.githubusercontent.com/render/math?math=\color{white}Z^{[l]}"> and <img src="https://render.githubusercontent.com/render/math?math=\color{white}A^{[l]}">:
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
| ReLU        |  (0, <img src="https://render.githubusercontent.com/render/math?math=\color{white}\infty">)   |  derivative of slope is 1 when z > 0,  is 0 when z < 0  | 
| Leasky ReLU  |  (-<img src="https://render.githubusercontent.com/render/math?math=\color{white}\infty">, <img src="https://render.githubusercontent.com/render/math?math=\color{white}\infty">)    |



**Why non-linear activation functions?** 

> If we use linear activation functions, no matter how many layers you have, the NN is just computing a linear function. 


**Why do we usually initialize <img src="https://render.githubusercontent.com/render/math?math=\color{white}W"> as small random values?** 

> large W -> large Z (Z = WX + b) -> end up at the flat parts of Sigmoid function 
-> gradient will be small -> gradient descent will be slow -> learning will be slow
>
>If you're not using Sigmoid or Tanh activation functions, it is less of an issue. But note if you're doing a binary classification, the output layer will be a Sigmoid function. 

**What distribution should we draw <img src="https://render.githubusercontent.com/render/math?math=\color{white}W"> from?**

> Normal distirbution. 
>
> In Python we should use `np.random.randn` (normal distribution) instead of `np.random.rand` (uniform distribution). 

**Logistic regression’s weights w should be initialized randomly rather than to all zeros, because if you initialize to all zeros, then logistic regression will fail to learn a useful decision boundary because it will fail to “break symmetry”, True/False?**


> False. Logistic Regression doesn't have a hidden layer. If you initialize the weights to zeros, the first example x fed in the logistic regression will output zero but the derivatives of the Logistic Regression depend on the input x (because there's no hidden layer) which is not zero. 
> 
> So at the second iteration, the weights values follow x's distribution and are different from each other if x is not a constant vector. 
>
> But in deep learning we should randomly initialize either <img src="https://render.githubusercontent.com/render/math?math=\color{white}W"> or <img src="https://render.githubusercontent.com/render/math?math=\color{white}b"> to "break symmetry". 
If both <img src="https://render.githubusercontent.com/render/math?math=\color{white}W"> and <img src="https://render.githubusercontent.com/render/math?math=\color{white}b"> values zero, <img src="https://render.githubusercontent.com/render/math?math=\color{white}A^{[1]}"> will be 0 because *tanh(0)=0*. 
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

| Regularization | Penalize | <img src="https://render.githubusercontent.com/render/math?math=\color{white}W"> | Feature selection | 
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

> A partial solution: force the variance of <img src="https://render.githubusercontent.com/render/math?math=\color{white}W"> to be constant and smaller. A recommended value is <img src="https://render.githubusercontent.com/render/math?math=\color{white}\frac{1}{n}"> but it depends on the activation function.  

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

><img src="https://render.githubusercontent.com/render/math?math=\color{white}v_t = \beta v_{t-1} + (1-\beta) \theta_t">
>
> <img src="https://render.githubusercontent.com/render/math?math=\color{white}v^{corrected}_t = \frac{v_t}{1-\beta^t}">

</details>

<br />

**What is momentum?**

> Momentum is a method to dampen down the changes in gradients and accelerate gradients vectors in the right direction using exponentially weighted averages. 

**Which of these is NOT a good learning rate decay scheme? Here, t is the epoch number.**

1. <img src="https://render.githubusercontent.com/render/math?math=\color{white}\alpha=0.95^t\alpha_0">
2. <img src="https://render.githubusercontent.com/render/math?math=\color{white}\alpha=e^t\alpha_0">
3. <img src="https://render.githubusercontent.com/render/math?math=\color{white}\alpha=\frac{1}{1+2*t}\alpha_0">
4. <img src="https://render.githubusercontent.com/render/math?math=\color{white}\alpha=\frac{1}{\sqrt{t}}\alpha_0">

<details>
    <summary>Click to see answer</summary>

> <img src="https://render.githubusercontent.com/render/math?math=\color{white}\alpha=e^t\alpha_0"> explodes <img src="https://render.githubusercontent.com/render/math?math=\color{white}\alpha"> instead of decaying it

</details>

<br />

**What is the process of parameter update in Adam?**

<details>
    <summary>Click to see answer</summary>

> <img src="https://render.githubusercontent.com/render/math?math=\color{white}v_{dW^{[l]}} = \beta_1 v_{dW^{[l]}} + (1-\beta_1) \frac{\partial J}{\partial W^{[l]}}">
> 
> <img src="https://render.githubusercontent.com/render/math?math=\color{white}v_{dW^{[l]}} ^{correted} = \frac{v_{dW^{[l]}}}{1 - (\beta_1)^t}">
>
> <img src="https://render.githubusercontent.com/render/math?math=\color{white}s_{dW^{[l]}} = \beta_2 v_{dW^{[l]}} + (1-\beta_2) (\frac{\partial J}{\partial W^{[l]}}) ^2">
>
> <img src="https://render.githubusercontent.com/render/math?math=\color{white}s_{dW^{[l]}} ^{correted} = \frac{s_{dW^{[l]}}}{1 - (\beta_2)^t}">
> 
> <img src="https://render.githubusercontent.com/render/math?math=\color{white}W^{[l]} = W^{[l]} - \alpha \frac{v^{corrected}_{dW^{[l]}}}  {\sqrt{s_{dW^{[l]}} ^{correted}} + \varepsilon}">

</details>

<br />

### Week 3: Hyperparameter Tuning, Batch Normalization and Programming Frameworks


**Why batch normalization?**

> Normalization can make training faster and hyperparameters more robust. 
>
> Values of each hidden layer are changing all the time because of changes in <img src="https://render.githubusercontent.com/render/math?math=\color{white}W"> and <img src="https://render.githubusercontent.com/render/math?math=\color{white}b">, suffering from the problem of covariance shift. Batch normalization guarantees the mean and variance of features of each layer (e.g., <img src="https://render.githubusercontent.com/render/math?math=\color{white}Z^{[2]}_1">, <img src="https://render.githubusercontent.com/render/math?math=\color{white}Z^{[2]}_2">) keep the same no matter how actual values of each node changes. 
> It allows each layer to learn by itself (more independently than no batch normalization), and speed up learning. 
>
> Mean and variance are governed by two learnable parameters <img src="https://render.githubusercontent.com/render/math?math=\color{white}\gamma"> and <img src="https://render.githubusercontent.com/render/math?math=\color{white}\beta">. Adding <img src="https://render.githubusercontent.com/render/math?math=\color{white}\gamma"> and <img src="https://render.githubusercontent.com/render/math?math=\color{white}\beta"> is because we don't want all the layers have the same mean and variance (mean = 0, variance = 1).


**Batch normalization fomula?**

<details>
    <summary>Click to see answer</summary>

> <img src="https://render.githubusercontent.com/render/math?math=\color{white}z_{norm}^{(i)} = \frac{z^{(i) - \mu}}{\sqrt{\sigma^2 + \epsilon}}">

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
| learning rate <img src="https://render.githubusercontent.com/render/math?math=\color{white}\alpha"> | <img src="https://render.githubusercontent.com/render/math?math=\color{white}r\in [-4,0], \alpha=10^r"> | 
| momentum <img src="https://render.githubusercontent.com/render/math?math=\color{white}\beta"> | around 0.9 | 
| mini-batch size | <img src="https://render.githubusercontent.com/render/math?math=\color{white}2^n"> | 
| # of hidden units | - |
| learning rate decay | <img src="https://render.githubusercontent.com/render/math?math=\color{white}10^r"> |
| # of layers <img src="https://render.githubusercontent.com/render/math?math=\color{white}L"> | - | 
| batch normalization <img src="https://render.githubusercontent.com/render/math?math=\color{white}\beta_1">, <img src="https://render.githubusercontent.com/render/math?math=\color{white}\beta_2">, <img src="https://render.githubusercontent.com/render/math?math=\color{white}\epsilon"> | 0.9, 0.99, <img src="https://render.githubusercontent.com/render/math?math=\color{white}10^{-8}"> | 


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
| <img src="https://render.githubusercontent.com/render/math?math=\color{white}f^{[l]}"> |  filter size |
| <img src="https://render.githubusercontent.com/render/math?math=\color{white}p^{[l]}"> | padding |
| <img src="https://render.githubusercontent.com/render/math?math=\color{white}s^{[l]}"> | stride | 
| <img src="https://render.githubusercontent.com/render/math?math=\color{white}n_c^{[l]}"> | number of filters | 

| Metric      | Dimension | 
| ----------- | ----------- | 
| Filter |  <img src="https://render.githubusercontent.com/render/math?math=\color{white}(f^{[l]}, f^{[l]}, n_c^{[l]})"> |
| Activations | <img src="https://render.githubusercontent.com/render/math?math=\color{white}(n_H^{[l]}, n_W^{[l]}, n_c^{[l]})"> |
| Weights | <img src="https://render.githubusercontent.com/render/math?math=\color{white}(f^{[l]}, f^{[l]}, n_c^{[l-1]}, n_c^{[l]})"> | 
| bias | <img src="https://render.githubusercontent.com/render/math?math=\color{white}(1, 1, 1, n_c^{[l]})"> | 


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


**AlexNet**

> Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). [Imagenet classification with deep convolutional neural networks](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html). Advances in neural information processing systems, 25.


**VGG - 16**

> Simonyan, K., & Zisserman, A. (2014). [Very deep convolutional networks for large-scale image recognition](https://arxiv.org/pdf/1409.1556.pdf). arXiv preprint arXiv:1409.1556.

**ResNet**

> He, K., Zhang, X., Ren, S., & Sun, J. (2016). [Deep residual learning for image recognition](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf). In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

**Inceptionm Network**

> Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A. (2015). [Going deeper with convolutions](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html). In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1-9).


**MobileNet**

> Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., ... & Adam, H. (2017). [Mobilenets: Efficient convolutional neural networks for mobile vision applications](https://arxiv.org/pdf/1704.04861.pdf). arXiv preprint arXiv:1704.04861.


**EfficientNet**

> Tan, M., & Le, Q. (2019, May). [Efficientnet: Rethinking model scaling for convolutional neural networks](http://proceedings.mlr.press/v97/tan19a/tan19a.pdf). In International conference on machine learning (pp. 6105-6114). PMLR.


**Suppose that in a MobileNet v2 Bottleneck block we have an <img src="https://render.githubusercontent.com/render/math?math=\color{white}n\times n \times 5"> input volume. We use 30 filters for the expansion. In the depthwise convolutions we use <img src="https://render.githubusercontent.com/render/math?math=\color{white}3 \times 3"> filters, and 20 filters for the projection.**

**How many parameters are used in the complete block, suppose we don't use bias?**

> Expansion filter: 5 * 30 = 150
> 
> Depthwise: 3 * 3 * 30 = 270 
> 
> Pointwise: 30 * 20 = 600
> 
> Total = 150 + 270 + 600 = 1020

**What does skip-connection do?**

> Skip-connections make it easy for the network to learn an identity mapping between the input and the output within the ResNet block. 


[⬆️ Back to top](#table-of-contents)

### Week 3: Detection Algorithms

**YOLO**

> Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). [You only look once: Unified, real-time object detection](https://arxiv.org/pdf/1506.02640.pdf). In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 779-788).

**What are advantages of YOLO algorithm?**

> - It can ouput more accurate bounding box
> - It's one single convolutional computation where you use one conv net with lots of shared computation between all the computation needed for all the cells. Therefore it's a very efficient algorithm. 

**How to evaluate object localization?**

> <img src="https://render.githubusercontent.com/render/math?math=\color{white}IoU = \frac{size\ of\ intersection}{size\ of\ union} ">

**How does non-max suppression work?**

> While BoundingBoxes: 
>  
> - Pick the box with the highest <img src="https://render.githubusercontent.com/render/math?math=\color{white}P_c">, add it to ouput list 
> - Calculate IoU of all the other boxes with the one in the last step. Discard boxes with IoU GREATER than sthreshold


**U-net**
> Ronneberger, O., Fischer, P., & Brox, T. (2015, October). [U-net: Convolutional networks for biomedical image segmentation](https://arxiv.org/pdf/1505.04597.pdf). In International Conference on Medical image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.

### Week 4: Face Recognition

**Siamese Network**

> Taigman, Y., Yang, M., Ranzato, M. A., & Wolf, L. (2014). [Deepface: Closing the gap to human-level performance in face verification](https://openaccess.thecvf.com/content_cvpr_2014/papers/Taigman_DeepFace_Closing_the_2014_CVPR_paper.pdf). In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1701-1708).

**Triplet Loss**
> Schroff, F., Kalenichenko, D., & Philbin, J. (2015). [Facenet: A unified embedding for face recognition and clustering](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf). In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 815-823).