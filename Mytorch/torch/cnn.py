from ..nn.sequential import Sequential
from ..nn.linear import Linear
from ..nn.activation_function import *
from ..nn.module import Module
from ..nn.maxpool import Maxpool
from ..nn.conv2d import Conv2d
from ..nn.activation_function import *


def CNN_base():
    # 初始化各层及激活函数
    # input 1*28*28
    conv1 = Conv2d("conv1",intputChannal=1,outputChnnal=3,stride=1,padding=0,kernel_size=(5,5)) # 24*24*3
    f1 = Relu("f1")
    conv2 = Conv2d("conv2",intputChannal=3,outputChnnal=6,stride=1,padding=0,kernel_size=(5,5))  # 20*20*6
    f2 = Relu("f2")
    linear1 = Linear("linear1", 2400, 128) 
    f3 = Relu("f3")
    linear2 = Linear("linear2", 128, 10)
    f4 = SoftMax("f4")


    # 构建网络
    model = Sequential("model")
    model.add_module(conv1)
    model.add_module(f1)
    model.add_module(conv2)
    model.add_module(f2)
    model.add_module(linear1)
    model.add_module(f3)
    model.add_module(linear2)
    model.add_module(f4)
    return model

def CNN():
    # 初始化各层及激活函数
    conv1 = Conv2d("conv1",intputChannal=1,outputChnnal=3,stride=1,padding=1,kernel_size=(3,3))
    f1 = Relu("f1")
    maxpool1 = Maxpool("maxpoo1", size = 2, stride = 2)
    conv2 = Conv2d("conv2",intputChannal=3,outputChnnal=6,stride=1,padding=1,kernel_size=(3,3)) 
    f2 = Relu("f2")
    maxpool2 =Maxpool("maxpoo2", size = 2, stride = 2)
    linear1 = Linear("linear1", 7 * 7 * 6, 128) 
    f3 = Relu("f3")
    linear2 = Linear("linear2", 128, 10)
    f4 = SoftMax("f4")


    # 构建网络
    model = Sequential("model")
    model.add_module(conv1)
    model.add_module(maxpool1)
    model.add_module(f1)
    model.add_module(conv2)
    model.add_module(maxpool2)
    model.add_module(f2)
    model.add_module(linear1)
    model.add_module(f3)
    model.add_module(linear2)
    model.add_module(f4)
    return model