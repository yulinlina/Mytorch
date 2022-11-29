#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：MyTorch 
@File    ：test_Sequential.py
@Author  ：尤敬斌
@Date    ：2022/11/3 23:51 
'''

import unittest

import numpy as np
from ..tensor import *
from ..nn.sequential import Sequential
from ..nn.linear import Linear
from ..nn.activation_function import Relu, Sigmoid, SoftMax

class SequentialTest(unittest.TestCase):

    def test_linear_forward1(self):
        batch_size = 5
        num_in, num_out = 5, 10

        # 初始化各层及激活函数
        sequential = Sequential("sequential")
        linear1 = Linear("linear1", num_in, 8)
        relu1 = Relu("relu1")
        linear2 = Linear("linear2", 8, 16)
        relu2 = Relu("relu2")
        linear3 = Linear("linear3", 16, num_out)
        relu3 = Relu("relu3")

        # 构建网络
        sequential.add_module(linear1)
        sequential.add_module(relu1)
        sequential.add_module(linear2)
        sequential.add_module(relu2)
        sequential.add_module(linear3)
        sequential.add_module(relu3)

        # 前向计算
        input = Tensor(np.zeros((batch_size, num_in)))
        output = sequential.forward(input)
        self.assertEqual(output.shape, (batch_size, num_out))


        # 显示网络结构
        sequential.show()



    def test_linear_forward2(self):
        batch_size = 5
        num_in, num_out = 5, 10

        # 初始化各层及激活函数
        sequential = Sequential("sequential")
        linear1 = Linear("linear1", num_in, 8)
        f1 = Relu("f1")
        linear2 = Linear("linear2", 8, 16)
        # f2 = Sigmoid("f2")
        f2 = Relu("f2")
        linear3 = Linear("linear3", 16, num_out)
        f3 = SoftMax("f3")

        # 构建网络
        sequential.add_module(linear1)
        sequential.add_module(f1)
        sequential.add_module(linear2)
        sequential.add_module(f2)
        sequential.add_module(linear3)
        sequential.add_module(f3)

        # 前向计算
        input = Tensor(np.zeros((batch_size, num_in)))
        output = sequential.forward(input)
        self.assertEqual(output.shape, (batch_size, num_out))

        # 显示网络结构
        sequential.show()



if __name__ == '__main__':
    unittest.main()  # 运行所有的测试用例