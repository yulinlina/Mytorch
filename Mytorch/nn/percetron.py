#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：MyTorch
@Author  ：njw1123
@Date    ：2022/11/4 12:33
'''

import  numpy as np

class Percetron():
    def __init__(self):
        self.eta = 1
        self.w = None
        self.b = None

    def train(self, train_x, train_y, eta = 1):
        # 初始化参数
        self.w = np.zeros(train_x.shape[1])
        self.b = 0
        self.eta = eta

        suc = False
        cnt = 0
        while not suc:
            cnt += 1
            if(cnt >= 100000):
                break
            suc = True
            for id in range(len(train_x)):
                x = train_x[id]
                y = train_y[id]
                # 如果满足条件则说明分类不正确
                if y * (x.dot(self.w) + self.b) <= 0:
                    # 对参数w, b进行更新
                    self.w = self.w + eta * y * x
                    self.b = self.b + eta * y
                    # 存在未正确分类的, suc设置为False
                    suc = False
        # 如果达到迭代上限, 仍未将完成对所有数据的分类, 则说明可能是线性不可分的, 返回False
        if(cnt == 100000):
            return False, self.w, self.b
        return True, self.w, self.b