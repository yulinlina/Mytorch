#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：MyTorch
@Author  ：尤敬斌
@Date    ：2022/11/4 12:33
'''


import unittest

from ..nn.module import Module

class ModuleTest(unittest.TestCase):

    def test_get_params(self):
        module = Module("module")
        params = module.get_params()
        self.assertEqual(params, module.params)

    def test_get_module_name(self):
        module = Module("module")
        module_name = module.get_module_name()
        self.assertEqual(module_name, "module")



if __name__ == '__main__':
    unittest.main()  # 运行所有的测试用例