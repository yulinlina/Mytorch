#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：MyTorch 
@File    ：linear.py
@Author  ：尤敬斌
@Date    ：2022/11/3 23:37 
'''

import numpy as np
from .module import Module
from .parameter import Parameter
from ..tensor import *


class Linear(Module):

    def __init__(self, module_name: str, num_in, num_out):
        super(Linear, self).__init__(module_name)
        self.num_in = num_in
        self.num_out = num_out
        self.params.w_dict[module_name] = Tensor(np.zeros((num_in, num_out)), requires_grad=True)
        self.params.b_dict[module_name] = Tensor(np.zeros((1, num_out)), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        bound = np.sqrt(6. / (self.num_in + self.num_out))
        self.params.w_dict[self.module_name] = Tensor(np.random.uniform(-bound, bound, (self.num_in, self.num_out)), requires_grad=True)
        self.params.b_dict[self.module_name] = Tensor(np.random.uniform(-bound, bound, (1, self.num_out)), requires_grad=True)


    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        """
        前向计算
        w [num_in, num_out]
        b [1, num_out]
        :param inputs: 输入 [batch,in_num]
        :return: 输出 [batch,out_num]
        """
        w = self.params.w_dict[self.module_name]
        b = self.params.b_dict[self.module_name]
        inputs = inputs.reshape((inputs.shape[0], -1))
        z = Tensor.__matmul__(inputs, w) + b

        return z

