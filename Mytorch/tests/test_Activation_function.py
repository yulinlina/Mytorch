#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：MyTorch
@Author  ：尤敬斌
@Date    ：2022/11/4 12:33
'''

import unittest
from ..tensor import *
import numpy as np

from ..nn.activation_function import Activation_function,Relu,Sigmoid,SoftMax
from ..nn.module import Module
from ..nn.linear import Linear

class Activation_functionTest(unittest.TestCase):

    def test_relu(self):
        batch_size = 5
        num_in, num_out = 4, 8
        linear1 = Linear("linear1", num_in, num_out)
        x = Tensor(np.random.randn(batch_size, num_in))
        relu1 = Relu("relu1")
        z = linear1.forward(x)
        t = z.datas.copy()
        a = relu1.forward(z)
        self.assertEqual((a.datas == np.maximum(t, 0)).all(), True)

    # def test_sigmoid(self):
    #     batch_size = 5
    #     num_in, num_out = 4, 8
    #     linear1 = Linear("linear1", num_in, num_out)
    #     x = Tensor(np.random.randn(batch_size, num_in))
    #     f = Sigmoid("f")
    #     z = linear1.forward(x)
    #     t = z.datas.copy()
    #     a = f(z)
    #     self.assertEqual((a.datas == 1/(1+np.exp(t))).all(), True)

    # def test_softmax(self):
    #     batch_size = 5
    #     num_in, num_out = 4, 8
    #     linear1 = Linear("linear1", num_in, num_out)
    #     x = Tensor(np.random.randn(batch_size, num_in))
    #     f = SoftMax("f")
    #     z = linear1.forward(x)
    #     t = z.datas.copy()
    #     t -= np.max(t)
    #     a = f(z)
    #     self.assertEqual((a.datas == np.exp(t) / np.sum(np.exp(t))).all(), True)

if __name__ == '__main__':
    unittest.main()
