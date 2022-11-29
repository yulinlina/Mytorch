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