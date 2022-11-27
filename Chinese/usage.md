---
sort: 2
---

# 快速入门
快速入门主要介绍如何使用Mytorch搭建神经网络和进行训练
## 模块导入
```  python
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.lines import Line2D

from Mytorch import nn, optim, tensor, loss
from Mytorch.nn.sequential import Sequential
from Mytorch.nn.linear import Linear
from Mytorch.nn.activation_function import *
from Mytorch.loss.lossFunction import CrossEntropy, MSE
from Mytorch.evaluator.performance import ClassificationPerformance
from Mytorch.optim.sgd import SGD
```
## 定义准确率函数
``` python
 def accuracy(a, y):
    size = a.shape[0]
    idx_a = np.argmax(a, axis=1)
    idx_y = np.argmax(y, axis=1)
    # cp = ClassificationPerformance(idx_a,idx_y)
    # return cp.getAccuracy()
    acc = sum(idx_a == idx_y)
    return acc
``` 
## 加载数据
``` python
m = loadmat("minst/mnist_small_matlab.mat")

trainData, train_labels = m['trainData'], m['trainLabels']
testData, test_labels = m['testData'], m['testLabels']
train_images = trainData.reshape(-1, 10000).transpose(1, 0)
train_labels = train_labels.transpose(1, 0)
test_images = testData.reshape(-1, 2000).transpose(1, 0)
test_labels = test_labels.transpose(1, 0) 
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)
``` 
```
(10000, 784)
(10000, 10)
(2000, 784)
(2000, 10)
``` 
## 定义网络结构
``` python
# 初始化各层及激活函数
model = Sequential("sequential")
linear1 = Linear("linear1", 784, 256)
f1 = Relu("f1")
linear2 = Linear("linear2", 256, 100)
f2 = Relu("f2")
linear3 = Linear("linear3", 100, 10)
f3 = SoftMax("f3")

# 构建网络
model.add_module(linear1)
model.add_module(f1)
model.add_module(linear2)
model.add_module(f2)
model.add_module(linear3)
model.add_module(f3)
``` 