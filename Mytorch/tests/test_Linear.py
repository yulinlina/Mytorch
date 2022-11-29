#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：MyTorch 
@File    ：test_Linear.py
@Author  ：尤敬斌
@Date    ：2022/11/3 13:59 
'''


import unittest
from ..tensor import *
import numpy as np

from ..nn.linear import Linear

class LinearTest(unittest.TestCase):

    def test_get_params(self):
        linear1 = Linear("linear1", 5, 8)
        params = linear1.get_params()
        self.assertEqual(params, linear1.params)

    def test_get_name(self):
        linear1 = Linear("linear1", 5, 8)
        self.assertEqual("linear1", linear1.get_module_name())

    def test_forward(self):
        batch_size = 5
        num_in, num_out = 4, 8
        linear1 = Linear("linear1", num_in, num_out)
        w = linear1.params.w_dict["linear1"].datas
        b = linear1.params.b_dict["linear1"].datas
        x = np.random.randn(batch_size, num_in)
        z = x.dot(w) + b
        self.assertEqual(z.shape, (batch_size, num_out))
        self.assertEqual((z == linear1.forward(Tensor(x)).datas).all(), True)


if __name__ == '__main__':
    unittest.main()     # 运行所有的测试用例