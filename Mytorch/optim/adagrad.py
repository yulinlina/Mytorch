#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：MyTorch
@Author  ：njw1123
@Date    ：2022/11/4 12:33
'''

from ..tensor import *
from Mytorch import nn
from ..nn.linear import Linear
from ..nn.conv2d import Conv2d 
import math
import numpy as np

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
