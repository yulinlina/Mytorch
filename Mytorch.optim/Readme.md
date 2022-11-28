---
sort: 4
---

# Mytorch.optim

### 一、简介

神经网络的学习的目的就是寻找合适的参数，使得损失函数的值尽可能小。解决这个问题的过程为称为最优化。解决这个问题使用的算法叫做优化器, 常见的优化器有SGD, Momentum, AdaGrad, Adam等, 下面将会介绍并实现SGD, AdaGrad, Adam三种优化器.

### 二、SGD

#### 1.原理介绍

SGD 的想法就是沿着梯度的反方向前进一定距离。用数学的语言来描述的话可以写成下式：<img src="file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml4508\wps1.jpg" alt="img" style="zoom:50%;" /> 

这里面, W表示需要更新的权重, <img src="file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml4508\wps2.png" alt="img" style="zoom:50%;" />表示损失函数对W的梯度, <img src="file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml4508\wps3.png" alt="img" style="zoom:50%;" />表示学习率, <img src="file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml4508\wps4.png" alt="img" style="zoom:50%;" />表示用右边的值更新左边的值.

与批量梯度下降算法相比较, 每次只使用一个样本进行梯度下降对参数进行更新.

SGD 的优点就是简单，容易实现。但是其缺点就是低效，低效的原因有两大方面:

- 函数呈延伸状，梯度指向了谷底，可能会使得损失函数值不停的在震荡，如下图所示

- 当前位置的梯度很小, 导致损失值下降很慢, 甚至无法走出该区域, 陷入比较差的极小值点

<img src="file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml4508\wps5.jpg" alt="img" style="zoom:50%;" />

 

#### 2. 代码实现

```python
class SGD():
    def __init__(self, module: nn.sequential, lr=1e-3, lr_decay=0):
        self.module = module
        self.learning_rate = lr
        self.lr_decay = lr_decay
        self.epoch = 0

    def step(self):
        if self.lr_decay:
            self.epoch += 1
            self.learning_rate = self.learning_rate * (1 - self.lr_decay) ** (
                    self.epoch // 100)
        else:
            self.learning_rate = self.learning_rate * 1

        for module_name, layer in self.module.module_dict.items():
            if isinstance(layer, Linear) == True or isinstance(layer, Conv2d) == True:
                layer.params.w_dict[module_name].datas -= self.learning_rate * layer.params.w_dict[module_name].grad
                layer.params.b_dict[module_name].datas -= self.learning_rate * layer.params.b_dict[module_name].grad

    def zero_grad(self):
        for module_name, layer in self.module.module_dict.items():
            if isinstance(layer, Linear) == True or isinstance(layer, Conv2d) == True:
                layer.params.w_dict[module_name].zero_grad()
                layer.params.b_dict[module_name].zero_grad()
```



#### 3. 结果展示

![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml4508\wps6.jpg) 

将深度学习引论课程中提供小型mnist数据集作为训练数据, 构建全连接神经网络, 经过十个epoch训练后, 可以发现模型的参数可以得到有效的优化.

### 三、AdaGrad

#### 1.原理介绍

这种方法主要是为了解决 SGD 低效的第二个原因,我们知道超参数学习率是一个很重要的参数，不同的参数对学习结果的影响很大，如果设置的值较小，会导致学习花费较多的时间，学习率大了就会导致学习发散而不能正常的进行。而AdaGrad采用了一种动态调整学习率的策略. 数学的描述如下：

<img src="file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml4508\wps7.jpg" alt="img" style="zoom:50%;" />

 

这里的<img src="file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml4508\wps8.png" alt="img" style="zoom: 67%;" />表示之前累计的梯度的平方和, 在之后更新梯度的时候, 使用学习率<img src="file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml4508\wps9.png" alt="img" style="zoom: 67%;" />除以<img src="file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml4508\wps10.png" alt="img" style="zoom:67%;" />, 并且为了防止除0, 所以增加了eps, 在这种方式下, 在训练开始的阶段学习率比较大, 容易跳出当前极小值, 走到更优的极小值点附近, 随着<img src="file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml4508\wps11.png" alt="img" style="zoom:67%;" />的累加, 学习率逐渐变小, 模型收敛到附件的极小值点, 但是这也导致了随着<img src="file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml4508\wps12.png" alt="img" style="zoom:67%;" />的累加, 学习率逐渐收敛到0, 但是模型可能并没有收敛的问题.

#### 2.代码实现

```python
class AdaGrad():
    def __init__(self, module, lr = 1e-2, eps = 1e-8):
        self.module = module; self.lr = lr; self.eps = eps
        self.weightm = []; self.biasm = []; 
        for module_name, layer in self.module.module_dict.items():
            self.weightm.append[{}]; self.biasm = [{}]; 
            weight = layer.params.w_dict[module_name]; bias = layer.params.b_dict[module_name]
            self.weightm[module_name] = np.zeros(weight.grad.shape)
            self.biasm[module_name] = np.zeros(bias.grad.shape)

    def step(self):
        for module_name, layer in self.module.module_dict.items():
            self.weightm[module_name] += layer.params.w_dict[module_name].grad ** 2
            self.biasm[module_name] += layer.params.b_dict[module_name].grad ** 2
            layer.params.w_dict[module_name].datas -= self.learning_rate * layer.params.w_dict[module_name].grad / (np.sqrt(self.weightm[module_name]) + self.eps)
            layer.params.b_dict[module_name].datas -= self.learning_rate * layer.params.b_dict[module_name].grad / (np.sqrt(self.biasm[module_name]) + self.eps)

    def zero_grad(self):
        for module_name, layer in self.module.module_dict.items():
            if isinstance(layer, Linear) or isinstance(layer, Conv2d) == True:
                layer.params.w_dict[module_name].zero_grad()
                layer.params.b_dict[module_name].zero_grad()
```



#### 3.结果展示

<img src="file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml4508\wps13.jpg" alt="img" style="zoom:80%;" />

 

经过十个epoch训练后, 可以发现模型的参数可以得到有效的优化.

### 四、Adam

#### 1.原理介绍

为了解决 SGD 低效的第一个原因, 动量被引入到了优化器中, 简单而言, 即当前参数更新的方向不仅依靠当前点的梯度方向, 而且参考了之前梯度方向.

Adam不进借鉴了动量方法, 而且引入 AdaGrad的自适应学习率, 并在此基础上对mt和vt进行了偏差纠正. 最终表达式如下所示:

<img src="file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml4508\wps14.jpg" alt="img" style="zoom:50%;" />

 

#### 2.代码实现

```python
class Adam():
    def __init__(self, module, beta1 = 0.9, beta2 = 0.99, lr = 1e-2, eps = 1e-8):
        self.module = module; self.learning_rate = lr; self.eps = eps
        self.beta1 = beta1; self.beta2 = beta2
        self.beta1t = beta1; self.beta2t = beta2
        self.weightv = {}; self.biasv = {}; 
        self.weightm = {}; self.biasm = {}; 
        self.t = 0
        for module_name, layer in self.module.module_dict.items():
            if isinstance(layer, Linear) or isinstance(layer, Conv2d) == True:
                weight = layer.params.w_dict[module_name]; bias = layer.params.b_dict[module_name]
                self.weightv[module_name] = np.zeros(weight.grad.shape)
                self.biasv[module_name] = np.zeros(bias.grad.shape)
                self.weightm[module_name] = np.zeros(weight.grad.shape)
                self.biasm[module_name] = np.zeros(bias.grad.shape)


    def step(self):
        for module_name, layer in self.module.module_dict.items():
            self.t += 1
            if isinstance(layer, Linear) or isinstance(layer, Conv2d) == True:
                lr =  self.learning_rate * math.sqrt(1.0 - self.beta2 ** self.t) / (1.0 - self.beta1**self.t)
                self.weightm[module_name] = self.weightm[module_name] * self.beta1 + layer.params.w_dict[module_name].grad * (1 - self.beta1)
                self.biasm[module_name] = self.biasm[module_name] * self.beta1 + layer.params.b_dict[module_name].grad * (1 - self.beta1)
                self.weightv[module_name] = self.weightv[module_name] * self.beta2 + layer.params.w_dict[module_name].grad ** 2 *  (1 - self.beta2)
                self.biasv[module_name] = self.biasv[module_name] * self.beta2 + layer.params.b_dict[module_name].grad ** 2 * (1 - self.beta2)

                layer.params.w_dict[module_name].datas -= lr * layer.params.w_dict[module_name].grad / (np.sqrt(self.weightv[module_name]) + self.eps)
                layer.params.b_dict[module_name].datas -= lr * layer.params.b_dict[module_name].grad / (np.sqrt(self.biasv[module_name]) + self.eps)
                self.beta1t = self.beta1 * self.beta1t
                self.beta2t = self.beta2 * self.beta2t

    def zero_grad(self):
        for module_name, layer in self.module.module_dict.items():
            if isinstance(layer, Linear) or isinstance(layer, Conv2d) == True:
                layer.params.w_dict[module_name].zero_grad()
                layer.params.b_dict[module_name].zero_grad()
```

#### 3. 结果展示

<img src="file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml4508\wps15.jpg" alt="img" style="zoom:80%;" />

 

在训练20个epoch后模型得到了有效的优化