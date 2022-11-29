
from ..nn.sequential import Sequential
from ..nn.linear import Linear
from ..nn.activation_function import *
from ..nn.module import Module

def MLP():
    model = Sequential("sequential")
    linear1 = Linear("linear1", 784, 256)
    f1 = Relu("f1")
    linear2 = Linear("linear2", 256, 100)
    f2 = Relu("f2")
    linear3 = Linear("linear3", 100, 10)
    f3 = SoftMax("f3")
    # 构建网络
    model.add_module(linear1)
    model.add_module(f1)
    model.add_module(linear2)
    model.add_module(f2)
    model.add_module(linear3)
    model.add_module(f3)
    return model
        





   