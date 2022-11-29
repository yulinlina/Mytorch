#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：project637755-94734 
@File    ：flatten.py
@Author  ：尤敬斌
@Date    ：2022/11/26 13:54 
'''


import numpy as np
from .module import Module
from ..tensor import *


class Flatten(Module):
    def __init__(self, module_name: str):
        super(Flatten, self).__init__(module_name)

    def forward(self, inputs: Tensor):
        assert isinstance(inputs, Tensor)
        batch_size = inputs.datas.shape[0]
        inputs = Tensor.reshape(inputs, (batch_size, -1))
        return inputs

