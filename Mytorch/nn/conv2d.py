#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：MyTorch
@Author  ：尤敬斌
@Date    ：2022/11/4 12:33
'''

import numpy as np
from .module import Module
from ..tensor import *


import numpy as np
from .module import Module
from ..tensor import *


class Conv2d(Module):
    """see help https://yulinlina.github.io/Mytorch/Mytorch.nn/Conv2d.html"""
    def __init__(self, module_name, intputChannal, outputChnnal, kernel_size, stride=1, padding=0):
        super(Conv2d, self).__init__(module_name)
        self.intputChannal = intputChannal
        self.outputChnnal = outputChnnal
        self.kernelRow = kernel_size[0]
        self.kernelCol = kernel_size[1]
        self.stride = stride
        self.paddingRow = self.paddingCol = padding

        self.params.w_dict[module_name] = Tensor(np.zeros((self.outputChnnal, self.intputChannal,  self.kernelRow, self.kernelCol)),requires_grad = True)
        self.params.b_dict[module_name] = Tensor(np.zeros((1, outputChnnal)),requires_grad = True)
        self.reset_parameters()

    def reset_parameters(self):
        bound = np.sqrt(6. / (self.outputChnnal + self.intputChannal))
        self.params.w_dict[self.module_name] = Tensor(np.random.uniform(-bound, bound, (self.outputChnnal, self.intputChannal,self.kernelRow, self.kernelCol)),requires_grad = True)
        self.params.b_dict[self.module_name] = Tensor(np.random.uniform(-bound, bound, (1, self.outputChnnal)),requires_grad = True)

    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        out = Tensor.__conv__(inputs,self.params.w_dict[self.module_name],self.params.b_dict[self.module_name],self.paddingRow, self.paddingCol, self.kernelRow, self.kernelCol, self.stride)
        return out




# class Conv2D:
#     def __init__(self,in_channel:int, out_channel:int,kernel_size,stride=1,padding=0):
#         self.input_channel = in_channel
#         self.output_channel =out_channel
#         if isinstance(kernel_size, int):
#             self.kernel_size_h = self.kernel_size_w = kernel_size
#         else:
#             assert len(kernel_size) == 2
#             self.kernel_size_h = kernel_size[0]
#             self.kernel_size_w = kernel_size[1]
#
#         if isinstance(padding, int):
#             self.padding_h = self.padding_w = padding
#         else:
#             assert len(kernel_size) == 2
#             self.padding_h = padding[0]
#             self.padding_w = padding[1]
#
#         self.stride = stride
#         self.padding =padding
#
#         self.kernel = np.random.randn(out_channel,in_channel,self.kernel_size_h,self.kernel_size_w)
#         self.bias = np.zeros((1,out_channel))
#
#     def forward(self,x):
#         """ Return the result(output_channels,out_h,out_h) of convolution
#
#         The reason why we reshape the weight of kernel is that we want each column represents each channel outcome for sequential listed result of cross correlation.
#         For example:
#             Column with index zero are ordered by batch_size and the first number of outsize values are produced by the first simple in the batch with the first channels.
#             Column with index one  are ordered by batch_size and the first number of outsize values are produced by the first simple in the batch with the second channels
#         """
#         self.batch_size,self.channels,self.height,self.width = x.shape
#         if self.padding:
#             x=self.pad(x)
#         self.outsize = self.out_w * self.out_h
#
#         input_mat = self.img2mat(x)
#         kernel_vec = self.kernel.reshape(self.output_channel, -1)
#         out = np.dot(input_mat, kernel_vec.T) + self.bias
#
#         return out.T.reshape(self.batch_size, self.out_h, self.out_w, -1).transpose(0, 3, 1, 2)
#
#     def pad(self,img):
#         self.out_h = (self.height + 2 * self.padding_h - self.kernel_size_h) // self.stride + 1
#         self.out_w = (self.width+ 2 * self.padding_w - self.kernel_size_w) // self.stride + 1
#         img = np.pad(img, [(0, 0), (0, 0),
#                            (self.padding_h, self.padding_h),
#                            (self.padding_w, self.padding_w)],
#                      'constant')
#         return img
#
#     def img2mat(self,img):
#         """ Return the matrix with each row that will be dotted by kernel_weight flatten and number of batch_size*outsize rows
#
#         In the for loop below,we do a slice to get a sub_image with shape (kernel_h,kernel_w) for all channels in the whole batch and reshape it as  (batch_size,channels*sub_image_shape) i.e.sub_image_shape=kernel_h*kernel_w
#         As the first number of outsize row in mat will the first sample output ,all the batch output will be insert over number of outsize rows.
#         The reason is explained below:
#             Row of mat represents the input sub_image of all channels(i.e channels=3 for RGB image) flatten together.
#             Each row will be dotted by the weights flatten accordingly as they are both the vector.
#             It is batch_size*out_windows_shape of values produced by the cross-correlation operation.
#         """
#         mat = np.zeros((self.batch_size * self.out_h * self.out_w, self.channels * self.kernel_size_h * self.kernel_size_w))
#         for y in range(self.out_h):
#             y_start = y * self.stride
#             y_end = y_start + self.kernel_size_h
#             for x in range(self.out_w):
#                 x_start = x * self.stride
#                 x_end = x_start + self.kernel_size_w
#                 mat[y * self.out_w + x::self.outsize, :] = img[:, :, y_start:y_end, x_start:x_end].reshape(self.batch_size, -1)
#         return mat
#
#
#
#
#
#
#
# x_test = np.random.randn(10,3,32,32)
# conv2=Conv2D(3,4,kernel_size = (3,3),padding=1)
# print(conv2.forward(x_test).shape)


