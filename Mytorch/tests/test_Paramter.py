#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：MyTorch
@Author  ：尤敬斌
@Date    ：2022/11/4 12:33
'''


import unittest


import numpy as np

from ..nn.parameter import Parameter


class ParamterTest(unittest.TestCase):

    def test_train_mode(self):
        param = Parameter()
        mode = param.train()
        self.assertEqual(mode, True)

    def test_update_w(self):
        param = Parameter()
        param.train()
        param.w_dict["conv1"] = np.zeros((5, 5))
        param.w_grad_dict["conv1"] = np.ones((5, 5))
        param.update_w(layer_name="conv1", learning_rate=1)
        self.assertEqual((param.w_dict["conv1"] == (-np.ones((5, 5)))).all(), True)


    def test_add(self):
        param1 = Parameter()
        param2 = Parameter()
        param3 = Parameter()
        param1.w_dict["fc1"] = np.zeros((2, 5))
        param2.w_dict["fc2"] = np.zeros((5, 8))
        param3.add(param1)
        param3.add(param2)
        self.assertEqual(len(param3.w_dict.keys()), 2)

if __name__ == '__main__':
    unittest.main()  # 运行所有的测试用例
