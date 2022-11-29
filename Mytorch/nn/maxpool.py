#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：MyTorch
@Author  ：王昊霖
@Date    ：2022/11/4 12:33
'''

import  numpy as np
from .module import Module
from ..tensor import *

class Maxpool(Module):
    def __init__(self, module_name, size, stride=1):
        super(Maxpool, self).__init__(module_name)
        self.size = size  # maxpool框的尺寸
        self.stride = stride

    def __call__(self, inputs, mode=True):
        return self.forward(inputs, mode)

    def forward(self, inputs, mode=True):
        # self.inputs = inputs
        return Tensor.__maxpool__(inputs, self.size, self.stride)

# class MaxPooling:
#     def __init__(self,size):
#         self.size=size
#
#     def forward(self,x):
#         self.batch_size, self.channels, self.height, self.width = x.shape
#         out = np.zeros((self.batch_size,self.channels,self.height//self.size,self.width//self.size))
#         for i in range(0,self.height//self.size):
#             for j in range(0, self.width // self.size):
#                 out[:,:,i,j] =np.max(x[:,:,i:(i+1)*self.size  ,j:(j+1)*self.size],axis=(2,3))
#         return out

# x_test = np.random.randn(10,3,32,32)
# maxpooling =MaxPooling(2)
# print(maxpooling.forward(x_test).shape)