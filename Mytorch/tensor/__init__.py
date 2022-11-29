"""
see help in https://yulinlina.github.io/Mytorch/Mytorch.tensor/
"""
import numpy as np
import gc


# 添加边
def addEdges(datas, Tensors, grad_fns):
    preEdges = []
    requires_grad = False
    for id, nod in enumerate(Tensors):
        requires_grad = requires_grad or nod.requires_grad
        if (nod.requires_grad):
            preEdges.append([nod, grad_fns[id]])
    return Tensor(datas, requires_grad=requires_grad, preEdges=preEdges)


class Tensor():
    def __init__(self, datas, requires_grad=False, preEdges=[], dtype=None, grad=None):
        self._datas = np.asarray(datas, dtype)
        self.requires_grad = requires_grad
        self.grad = np.zeros(self._datas.shape)
        self.preEdges = preEdges

    # 返回Tensor的数据
    @property
    def datas(self):
        return self._datas

    # 设置Tensor的数据
    @datas.setter
    def datas(self, new_val):
        self._datas = np.asarray(new_val)
        self.grad = np.zeros(new_val.shape)

    # 返回Tensor形状
    @property
    def shape(self):
        return self._datas.shape

    # 修改Tensor的形状
    def reshape(self, new_shape):
        shape = self.shape
        self._datas = self._datas.reshape(new_shape)

        def grad_fn(grad):
            return grad.reshape(shape)

        Tensors = []
        grad_fns = []
        Tensors.append(self)
        grad_fns.append(grad_fn)
        return addEdges(self._datas, Tensors, grad_fns)

    # 返回最大值
    @classmethod
    def __max__(self, *command, **map):
        return Tensor(np.max(*command, **map))

    # 清空梯度
    def zero_grad(self):
        self.grad = np.zeros(self.shape)

    # 进行反向传播
    def backward(self, grad=None):
        gradient = 1.0 if grad is None else grad
        tmp_grad = np.array(gradient)
        self.grad = self.grad + tmp_grad

        for nextTensor, grad_fn in self.preEdges:
            gradNew = grad_fn(tmp_grad)
            nextTensor.backward(gradNew)
            gc.collect()

    # 重载大于号
    def __gt__(self, obj):
        return self.datas > obj.datas

    # 重载加法
    def __add__(self, obj):
        datas = self.datas + obj.datas

        def grad_fn1(grad):
            grad = grad * 1.0
            if (grad.shape != self.shape):
                grad = grad.sum(axis=0)
            return grad

        def grad_fn2(grad):
            grad = grad * 1.0
            if (grad.shape != obj.shape):
                grad = grad.sum(axis=0)
            return grad

        Tensors = []
        grad_fns = []
        Tensors.append(self)
        grad_fns.append(grad_fn1)
        Tensors.append(obj)
        grad_fns.append(grad_fn2)
        return addEdges(datas, Tensors, grad_fns)
        # 重载减法

    def __sub__(self, obj):
        return self + (-obj)

    # 重载乘法
    def __mul__(self, obj):
        datas = self.datas * obj.datas

        def grad_fn1(grad):
            grad = grad * obj.datas
            return grad

        def grad_fn2(grad):
            grad = grad * self.datas
            return grad


        Tensors = []
        grad_fns = []
        Tensors.append(self)
        grad_fns.append(grad_fn1)
        Tensors.append(obj)
        grad_fns.append(grad_fn2)
        return addEdges(datas, Tensors, grad_fns)
        # 重载除法

    def __truediv__(self, obj):
        datas = self.datas / obj.datas

        def grad_fn1(grad):
            grad = grad / obj.datas
            return grad

        def grad_fn2(grad):
            grad = -grad / obj.datas ** 2
            return grad

        Tensors = []
        grad_fns = []
        Tensors.append(self)
        grad_fns.append(grad_fn1)
        Tensors.append(obj)
        grad_fns.append(grad_fn2)
        return addEdges(datas, Tensors, grad_fns)
        # 取反

    def __neg__(self):
        datas = -self.datas

        def grad_fn1(grad):
            return -grad

        Tensors = []
        grad_fns = []
        Tensors.append(self)
        grad_fns.append(grad_fn1)
        return addEdges(datas, Tensors, grad_fns)
        # 重载角标访问

    def __getitem__(self, key):
        datas = self.datas[key]

        def grad_fn1(grad):
            grads = np.ones_like(self.datas)
            grads[key] = grad
            return grads

        Tensors = []
        grad_fns = []
        Tensors.append(self)
        grad_fns.append(grad_fn1)
        return addEdges(datas, Tensors, grad_fns)
        # 重载矩阵乘法

    def __matmul__(self, obj):
        datas = self.datas.dot(obj.datas)

        def grad_fn1(grad):
            return grad @ obj.datas.T

        def grad_fn2(grad):
            return self.datas.T @ grad

        Tensors = []
        grad_fns = []
        Tensors.append(self)
        grad_fns.append(grad_fn1)
        Tensors.append(obj)
        grad_fns.append(grad_fn2)
        return addEdges(datas, Tensors, grad_fns)
        # # 返回数据长度

    def __len__(self):
        return len(self.datas)

    # 进行log运算
    def log(self):
        datas = np.log(self.datas + 1e-5)

        def grad_fn1(grad):
            return grad / (self.datas + 1e-5)

        Tensors = []
        grad_fns = []
        Tensors.append(self)
        grad_fns.append(grad_fn1)
        return addEdges(datas, Tensors, grad_fns)
        # 进行epx运算

    def exp(self):
        datas = np.exp(self.datas)

        def grad_fn1(grad):
            return grad * datas

        Tensors = []
        grad_fns = []
        Tensors.append(self)
        grad_fns.append(grad_fn1)
        return addEdges(datas, Tensors, grad_fns)
        # 进行softmax

    def __softmax__(self):
        return softmax__(self)
        # 求和

    def sum(self, axis=None, keepdims=False):
        return sum_(self, axis=axis, keepdims=keepdims)

    # 实现卷积操作
    def __conv__(self, TensorKernel, TensorBias, paddingRow, paddingCol, kernelRow, kernelCol, stride):
        return conv_(self, TensorKernel, TensorBias, paddingRow, paddingCol, kernelRow, kernelCol, stride)

    # 实现最大池化
    def __maxpool__(Tensorimg, sioute, stride):
        return maxpool_(Tensorimg, sioute, stride)


def softmax__(img):
    s = img.datas
    max = np.max(s)
    value = np.exp(s - max) / np.sum(np.exp(s - max), axis=1, keepdims=True)

    def grad_func_img(grad):
        sioute = value.shape[1]
        batch_sioute = value.shape[0]
        array = np.array([])
        for i in range(batch_sioute):
            array = np.append(array,
                              -np.dot(value[i:i + 1, :].T, value[i:i + 1, :]) + np.identity(sioute) * value[i:i + 1, :])
        array = array.reshape(batch_sioute, sioute, sioute)
        delta1 = np.matmul(np.expand_dims(grad, 1), array)
        delta1 = np.squeeze(delta1, 1)
        return delta1

    Tensors = []
    grad_fns = []
    Tensors.append(img)
    grad_fns.append(grad_func_img)
    return addEdges(value, Tensors, grad_fns)


def sum_(Tensor1, axis=None, keepdims=False):
    datas = Tensor1.datas.sum(axis=axis, keepdims=keepdims)

    def grad_fn1(grad):
        if axis is None or keepdims:
            return grad * np.ones_like(Tensor1.datas)
        elif type(axis) is int:
            return np.expand_dims(grad, axis=axis) * np.ones_like(Tensor1.datas)

    Tensors = []
    grad_fns = []
    Tensors.append(Tensor1)
    grad_fns.append(grad_fn1)
    return addEdges(datas, Tensors, grad_fns)


def im2col(img, kernelRow, kernelCol, stride=1, paddingRow=0, paddingCol=0):
    batchSioute, channels, imgRow, imgCol = img.shape

    outRow = (imgRow + 2 * paddingRow - kernelRow) // stride + 1
    outCol = (imgCol + 2 * paddingCol - kernelCol) // stride + 1
    img = np.pad(img, [(0, 0), (0, 0), (paddingRow, paddingCol), (paddingRow, paddingCol)], 'constant')
    col = np.zeros((batchSioute * outCol * outRow, channels * kernelCol * kernelRow))

    for idxRow in range(outRow):
        for idxCol in range(outCol):
            col[idxRow * outCol + idxCol:: outCol * outRow, :] = img[:, :, idxRow * stride:idxRow * stride + kernelRow,
                                                                 idxCol * stride: idxCol * stride + kernelCol].reshape(
                batchSioute, -1)
    return col


def col2img(col, kernelRow, kernelCol, outputShape, stride=1):
    batchSioute, channels, imgRow, imgCol = outputShape
    outRow = (imgRow - kernelRow) // stride + 1
    outCol = (imgCol - kernelCol) // stride + 1
    img = np.zeros(outputShape)

    for idxRow in range(outRow):
        for idxCol in range(outCol):
            img[:, :, idxRow * stride:idxRow * stride + kernelRow, idxCol * stride:idxCol * stride + kernelCol] += col[
                                                                                                                   idxRow * outCol + idxCol::outCol * outRow,
                                                                                                                   :].reshape(
                batchSioute, channels, kernelRow, kernelCol)
    return img

def conv_(TensorImg, TensorKernel, TensorBias, paddingRow, paddingCol, kernelRow, kernelCol, stride):
    img = TensorImg.datas
    kernel = TensorKernel.datas
    bias = TensorBias.datas
    batchSize, channals, imgRow, imgCol = img.shape
    outRow = (imgRow + 2 * paddingRow - kernelRow) // stride + 1
    outCol = (imgCol + 2 * paddingCol - kernelCol) // stride + 1
    imgTmp = im2col(img, kernelRow, kernelCol, stride, paddingRow, paddingCol)
    kernelTmp = im2col(kernel, kernelRow, kernelCol).T
    out = (np.dot(imgTmp, kernelTmp) + bias).reshape(batchSize, outRow, outCol, -1).transpose(0, 3, 1, 2)
    
    def grad_kernel(grad):
        grad = grad.reshape(out.shape)
        imgTmp = im2col(img, kernelRow, kernelCol, stride, paddingRow, paddingCol).reshape(batchSize, outRow * outCol,
                                                                                           channals,
                                                                                           kernelRow * kernelCol).transpose(
            2, 3, 1, 0).reshape(kernelRow * kernelCol * channals, outRow * outCol * batchSize).T
        grad_w = (grad.transpose(1, 2, 3, 0).reshape(out.shape[1], -1) @ imgTmp).reshape(kernel.shape)
        return grad_w

    def grad_bias(grad):
        grad = grad.reshape(out.shape)
        grad = np.sum(grad, axis=(0, 2, 3), keepdims=True)
        grad = grad.reshape(grad.shape[0], grad.shape[1])
        return grad

    def grad_func_img(grad):
        grad = grad.reshape(out.shape)
        flip_kernel = np.flipud(np.fliplr(np.flip(kernel))).transpose(1, 0, 2, 3)
        if stride > 1:
            delta = np.zeros((out.shape[0], out.shape[1], outRow, outCol))
            delta[:, :, ::stride, ::stride] = grad
            grad = delta

        grad = im2col(grad, kernelRow, kernelCol, 1, kernelRow - 1, kernelCol - 1)
        flip_kernel = im2col(flip_kernel, kernelRow, kernelCol).T
        delta = np.dot(grad, flip_kernel).reshape(img.shape[0], img.shape[2] + 2 * paddingRow,
                                                  img.shape[3] + 2 * paddingCol, channals).transpose(0, 3, 1, 2)
        delta = delta[:, :, paddingRow:delta.shape[2] - paddingRow, paddingCol:delta.shape[3] - paddingCol]
        return delta
    Tensors = []
    grad_fns = []    
    Tensors.append(TensorImg)
    grad_fns.append(grad_func_img)
    Tensors.append(TensorKernel)
    grad_fns.append(grad_kernel)
    Tensors.append(TensorBias)
    grad_fns.append(grad_bias)
    return addEdges(out, Tensors, grad_fns)


def maxpool_(Tensorimg, sioute, stride):
    img = Tensorimg.datas
    batchSioute, channels, imgRow, imgCol = img.shape
    outRow = (imgRow - sioute) // stride + 1
    outCol = (imgCol - sioute) // stride + 1
    imgTmp = im2col(img, sioute, sioute, stride).reshape(batchSioute * outRow * outCol * channels, sioute * sioute)
    out = np.max(imgTmp, axis=1, keepdims=True)
    index = np.argmax(imgTmp, axis=1)
    out = out.reshape(batchSioute, outRow, outCol, channels).transpose(0, 3, 1, 2)

    def grad_img(grad):
        grad = grad.reshape((batchSioute, channels, outRow, outCol)).transpose(0, 2, 3, 1).reshape(-1, 1)
        delta = np.zeros((batchSioute * outRow * outCol * channels, sioute * sioute))
        delta[range(delta.shape[0]), index] = grad.reshape(grad.shape[0])
        delta = delta.reshape(batchSioute * outRow * outCol, -1)
        delta = col2img(delta, sioute, sioute, img.shape, stride)
        return delta

    Tensors = []
    grad_fns = []
    Tensors.append(Tensorimg)
    grad_fns.append(grad_img)
    return addEdges(out, Tensors, grad_fns)