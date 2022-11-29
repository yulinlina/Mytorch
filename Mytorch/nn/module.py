from collections import OrderedDict

import numpy as np
from abc import ABCMeta, abstractmethod
from .parameter import Parameter

class Module:

    def __init__(self, module_name: str):
        """
        :param module_name: module_name是module的唯一标识,
        两个module的名字不能重复,通过module_name可以查询到module的所有参数信息
        """
        self.params = Parameter()
        self.module_name = module_name

    def get_params(self) -> Parameter:
        """
        获取module的全部参数信息
        :return: module的全部参数信息
        """
        return self.params

    def get_module_name(self) -> str:
        """
        获取模块名
        :return: 模块名
        """
        return self.module_name

    @abstractmethod
    def forward(self, input):
        """
        前向计算
        :param input: 输入
        :return: 输出
        """
        pass

    def zero_grad(self):
        """
        梯度清零
        :return:
        """
        pass

    def show(self):
        """
        显示网络结构
        """
        pass

    def train(self):
        self.params.train()

    def eval(self):
        self.params.eval()

    def __call__(self, inputs):
        return self.forward(inputs)





