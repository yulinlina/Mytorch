#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：MyTorch
@Author  ：王昊霖
@Date    ：2022/11/4 12:33
'''

import numpy as np
from ..tensor import *
from abc import ABCMeta, abstractmethod


class LossFunction:
    def __init__(self):
        self.loss = None

    def __call__(self, predictions, targets):
        return self.fit(predictions, targets)

    @abstractmethod
    def fit(self, predictions, targets):
        pass


class CrossEntropy(LossFunction):
    def fit(self, predictions, targets):
        assert predictions.shape == targets.shape
        self.loss = -Tensor.sum((targets * Tensor.log(predictions)))
        return self.loss



class MSE(LossFunction):
    def fit(self, predictions, targets):
        assert predictions.shape == targets.shape
        return Tensor.sum((predictions-targets)*(predictions-targets)) / targets.shape[0]


# class CrossEntropyLoss:
#     """
#     The input is expected to contain the unnormalized logits for each class (which do not need to be positive or sum to 1, in general).
#     We do softmax and negative-log likelihood together in this loss function
#     """
#     def __init__(self,reduction =True):
#         pass
#
#
#     def softmax(self,x):
#         """
#         :param x:shape(batch_size,out_features)
#         :return:
#         """
#         max_x = np.max(x,axis=1,keepdims=True)
#         x_exp = np.exp(x-max_x)
#         partition = np.sum(x_exp,axis=1,keepdims=True)
#         return  x_exp/partition
#
#
#
#     def forward(self,y_hat,y,epsilon=1e-7):
#         """
#         :param y_hat:  shape(N,C) for each sample ,it has to be a vector of size (C),
#         where C =number of classes(i.e.In SVHN dataset C=10).Each value is the probabilities for each class
#         :param y:containing class indices,shape(N,classes),for N = batch_size
#
#         :return: negative log likelihood
#         """
#         self.batch_size = y_hat.shape[0]
#         self.target = y
#         self.y_hat = self.softmax(y_hat)
#         return -np.log( self.y_hat[range(len(y_hat)),np.argmax(y,axis=1)]+epsilon).mean()
