---
sort: 3
---

## 抽象类 Activation_function

```python
import numpy as np
from ..tensor import *
from .module import Module

from abc import ABCMeta, abstractmethod


class Activation_function(Module):

    def __init__(self, module_name):
        super(Activation_function, self).__init__(module_name)
        self.params.z_dict[module_name] = None
        self.params.a_dict[module_name] = None

    def __call__(self, z):
        return self.forward(z)

    def forward(self, z):
        self.params.z_dict[self.module_name] = Tensor(z.datas.copy())
        a = self.f(z)
        self.params.a_dict[self.module_name] = Tensor(z.datas.copy())
        return a

    @abstractmethod
    def f(self, z):
        pass

```
## Relu
```python
class Relu(Activation_function):

    def f(self, z):
        return z * Tensor((z > Tensor(0)))

    # def df(self, z):
    #     x = z.copy()
    #     x[x <= 0] = 0
    #     return x
```
## Sigmoid
```python
class Sigmoid(Activation_function):

    def f(self, z):
        return 1 / (1 + Tensor.exp(-z))

    # def df(self, z):
    #     g = self.forward(z)
    #     return g * (1 - g)
```
## Softmax
```python
class SoftMax(Activation_function):

    def f(self, z):
        # z -= np.max(z, axis=1, keepdims=True)   # 为了稳定地计算softmax概率，减掉最大的那个元素
        # z = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
        return Tensor.__softmax__(z)

    # def df(self, z):
    #     x = self.f(z)
    #     s = x.reshape(-1, 1)
    #     return np.diagflat(s) - np.dot(s, s.T)
```
