#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：MyTorch
@Author  ：王昊霖
@Date    ：2022/11/4 12:33
'''
from tensor import  Tensor
class Node:
    def __init__(self):
        self.pre = [Node()]
        self.tensor= Tensor()
        self.function =None
        self.req_grad


