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