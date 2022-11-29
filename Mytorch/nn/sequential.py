#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：MyTorch 
@File    ：sequential.py
@Author  ：尤敬斌
@Date    ：2022/11/3 23:36 
'''

from collections import OrderedDict
from .module import Module
from .activation_function import Activation_function
from .conv2d import Conv2d
from .linear import Linear
from .maxpool import Maxpool

import schemdraw
from schemdraw.flow import *


class Sequential(Module):

    def __init__(self, module_name):
        """
        重写构造函数
        :param module_name:
        """
        super(Sequential, self).__init__(module_name)
        self.module_dict = OrderedDict()    # 用来按序存放所有的module

    def add_module(self, module):
        """
        在当前网络的最后一个module后面添加一个 module
        :param module: 新的 module
        :return:
        """
        self.module_dict[module.get_module_name()] = module
        self.params.add(module.params)

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, input):
        """
        遍历 self.module_dict 进行前向计算
        :param input:
        :return:
        """
        for module_name, module in self.module_dict.items():
            input = module.forward(input)
            if isinstance(module, Activation_function):
                self.params.add(module.params)
        return input

    def show(self):
        """
        显示网络结构
        :return:
        """
        # 展示网络结构（简易版）
        print("\n\n**********************************")
        print("network:", self.module_dict.keys())
        print("w_dict:", self.params.w_dict.keys())
        print("b_dict:", self.params.b_dict.keys())
        print("z_dict:", self.params.z_dict.keys())
        print("a_dict:", self.params.a_dict.keys())
        print("**********************************\n\n")

        with schemdraw.Drawing() as d:
            d += Start(w=5).label("input")
            input_shape = None
            for module_name, module in self.module_dict.items():
                shape = input_shape
                if isinstance(module, Linear):
                    shape = self.params.w_dict[module_name].shape[0]
                    input_shape = self.params.w_dict[module_name].shape[1]
                elif isinstance(module, Activation_function):
                    shape = input_shape
                elif isinstance(module, Maxpool):
                    shape = None
                elif isinstance(module, Conv2d):
                    shape = None

                if shape is None:
                    d += Arrow().down().label(f"{shape}")
                else:
                    d += Arrow().down().label(f"{shape}*1")

                extra = ""
                if isinstance(module, Conv2d):
                    extra += f"kernel_size={module.params.w_dict[module_name].shape},stride={module.stride},padding={module.paddingCol}"

                d += Process(w=20).label(f"{module_name} ({type(module).__name__}) {extra}")

            d += Arrow().down().label(f"{input_shape} * 1")
            d += Start(w=5).label("output")