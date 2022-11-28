---
sort: 3
---

# Mytorch.tensor

## 3.5 tensor

### 3.5.1 简介

tensor模块中定义了一个Tensor类, Tensor类中主要存放了一个多维数组, 并对该多维数组重载了基本运算, 例如修改形状, 进行矩阵相乘, 取log, 求和等.

在此基础上，为实现梯度的自动计算, 我们引入了计算图, 其实一个Tensor对象即为计算图的中的一个节点.

<img src="file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml15768\wps1.jpg" alt="img" style="zoom: 67%;" />

 

Tensor 属性

| datas:         | 用于存放该Tensor对象存储的变量值                             |
| -------------- | ------------------------------------------------------------ |
| grad:          | 存放对该节点变量的梯度                                       |
| requires_grad: | 表示该节点的变量是否需要梯度                                 |
| preEdges       | 存放该节点的前驱边, 其中每个边包含两个信息，一个是后继节点, 一个是对后继节点的求导函数 |
| dtype          | 表示该Tensor存放变量的数据类型                               |

 

Tensor 方法：

| datas()                                          | 获得datas的值, 对datas值的修改                  |
| ------------------------------------------------ | ----------------------------------------------- |
| shapes()                                         | 获取datas的形状, 对datas形状的修改              |
| zero_grad()                                      | 清空grad                                        |
| backward()                                       | 从该Tensor开始反向传播计算grad                  |
| __gt__() __add__() __sub__()__mul__()  __neg__() | 重载大于号,加法,减法,乘法，除法，取反，角标访问 |
| __matmul__()                                     | 矩阵乘法                                        |
| log()                                            | log                                             |
| sum()                                            | sum                                             |
| exp()                                            | exp                                             |
| softmax()                                        | softmax                                         |

 

### **3.5.2 计算图介绍**

梯度计算采用动态图机制，在模型进行前行计算的时候，会逐步建立出计算图，计算图的每个节点代表了一种变量，计算图之间的关系记录了各种变量之间的关系.

以$y=(a+b)*(b+c)$模拟计算图的计算过程. 在此之前, 我们假设a,b,c三个节点分别存放了a,b,c三个变量的值

前向计算部分:

建立f节点, a, b节点连接向f节点,f节点的值通过a+b获得, f与a,b节点的边上分别存放了f对a,b的求导函数

建立g节点, b, c节点连接向g节点,g节点的值通过b+c获得, g与b, c节点的边上分别存放了g对b, c的求导函数

建立y节点, f, g节点连接向y节点,y节点的值通过f*g获得, y与f, g节点的边上分别存放了y对f, g的求导函数

这样我们就完成了前向计算, 并且建立了计算图

梯度的反向传播:

将y节点部分的梯度初始化成1

y节点通过和f, g连接的边，以及边上的对f, g的求导函数, 将y节点的梯度传递到f, g节点

f节点通过和a, b连接的边，以及边上的对a, b的求导函数, 将f节点的梯度传递到a, b节点

g节点通过和b, c连接的边，以及边上的对b, c的求导函数, 将g节点的梯度传递到b, c节点, 值得注意的是, b节点之前已经获得了f节点传递过来的梯度, 所以这里需要在前面已经获得的梯度的基础上, 累计g节点传递过来的梯度

<img src="file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml15768\wps2.jpg" alt="img" style="zoom:80%;" />

 

通过上面的过程, 我们就可以获得了a, b, c三个节点的梯度了, 另外, 需要注意的是, 每个节点其实还有一个require_grad参数, 用来标识该节点是否需要梯度, 如果该节点需要梯度则会在反向传播的时候从他的后继节点传递来梯度.

### **3.5.3 逐步实现**

#### (1) Tensor的初始化

```python
def __init__(self, datas, requires_grad = False, preEdges = [], dtype = None, grad = None):
     self._datas = np.asarray(datas, dtype)
     self.requires_grad = requires_grad
     self.grad = np.zeros(self.shape)
     self.preEdges = preEdges
```

类内定义属性的含义在Tensor类的顶层设计中已经提及, 其中requires_grad默认设置成False, preEdges默认设置成空列表, grad默认设置成全零

#### (2) 创建新的Tensor,并添加前驱边

```python
def addEdges(datas, Tensors, grad_fns):
   preEdges = []
   requires_grad = False
   for id, nod in enumerate(Tensors):
       requires_grad = requires_grad or nod.requires_grad
       if(nod.requires_grad):
           preEdges.append([nod, grad_fns[id]])
   return Tensor(datas, requires_grad = requires_grad, preEdges = preEdges)


```

datas为新的节点的变量值, Tensors存放了新的节点的前驱节点, grad_fns存放了对于每个前驱节点的求导函数, 当存在前驱节点的变量需要求梯度时, 新创建的节点也需要求梯度, 这样才能反向传播回去.

#### (3) datas的获取及设置

```python
    # 返回Tensor的数据
    @property
    def datas(self):
        return self._datas

    # 设置Tensor的数据
    @datas.setter
    def datas(self, new_val):
        self._datas = np.asarray(new_val)
        self.grad = np.zeros(new_val.shape)
```

获取datas时, 直接返回datas即可, 在修改datas后, 记得将grad的清空

#### (4) datas的形状获取与改变

```python
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
```

获取形状时直接返回即可, 当datas的形状改变后, 需要创建新的新的节点, 记录改变后的变量值, 求导则是改变形状的逆操作, 即将grad改变后的形状变为改变前的形状.

#### (5) **获取长度**

```python
def __len__(self):
	return len(self.datas)
```

直接返回长度即可

#### (6) 清空梯度

```python
    # 清空梯度
    def zero_grad(self):
        self.grad = np.zeros(self.shape)
```

将grad变为_datas相同形状的全零数组即可

#### (7) **进行反向传播**

```python
    # 进行反向传播
    def backward(self, grad=None):
        gradient = 1.0 if grad is None else grad
        tmp_grad = np.array(gradient)
        self.grad = self.grad + tmp_grad

        for nextTensor, grad_fn in self.preEdges:
            gradNew = grad_fn(tmp_grad)
            nextTensor.backward(gradNew)
            gc.collect()
```

如果前面没有梯度传过来, 为None, 则我们将传递的grad初始化成1, 然后将梯度累加到当前节点上, 并遍历该节点的所有前驱边, 计算得到传递给前驱节点的梯度, 最后调用前驱节点的反向传播函数, 继续向前传播.

#### (8) **重载大于号**

```python
    # 重载大于号
    def __gt__(self, obj):
        return self.datas > obj.datas
```

直接调用numpy进行大于比较即可

#### (9) **重载加法**

```python
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
```

新建节点记录加法的结果, 并将两个输入作为前驱节点, 连接在新建节点后面, 边权为对两个输入的求导, 值得注意的是求导部分对grad的形状进行了调整，这是因为numpy中存在广播机制, 会导致在反向传播的过程中发生数组形状的不匹配, 所以需要进行调整.

#### (10) **重载取反**

```python
    def __neg__(self):
        datas = -self.datas

        def grad_fn1(grad):
            return -grad

        Tensors = []
        grad_fns = []
        Tensors.append(self)
        grad_fns.append(grad_fn1)
        return addEdges(datas, Tensors, grad_fns)
```

新建节点记录输入数据的取反, 其前驱边连接输入数据的节点, 对于原始节点的求导为grad取反

#### (11) **重载减法**

```python
    def __sub__(self, obj):
        return self + (-obj)
```

通过对加法和取反的重载, 可以直接实现对减法的重载

#### (12) **重载乘法**

```python
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
```

这里的相乘是指, 两个矩阵逐元素相乘, 新建节点记录两个输入相乘的结果, 其前驱边为连接输入数据的节点, 对于某个输入节点的求导为grad*另一个输入节点的变量值.

#### (13)重载除法

```python
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
```

假设两个输入节点为x, y, 建立新节点存储z=x/y, 并且z与x, y节点分别连接, z与x连接的边上存放z对x的求导1/y, z与y连接边上存储z对y的求导-1/y^2

#### (13) **重载矩阵乘法**

 ```python
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
 ```

假设两个输入节点为x, y, 建立新节点存储z=x.dot(y), 并且z与x, y节点分别连接, z与x连接的边上存放z对x的求导y.T, z与y连接边上存储z对y的求导x.T, 至于求导是左乘还是右乘, 根据矩阵的形状即可确定

#### (14) **重载角标访问**

```python
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
```

根据键值, 直接返回对应的变量值即可, 求导的话是创建一个和原始数组相同形状的全零矩阵, 只给对应角标的赋予梯度.

#### (15) 取exp

```python
    def exp(self):
        datas = np.exp(self.datas)

        def grad_fn1(grad):
            return grad * datas

        Tensors = []
        grad_fns = []
        Tensors.append(self)
        grad_fns.append(grad_fn1)
        return addEdges(datas, Tensors, grad_fns)
```

调用numpy实现的取exp函数, 计算得到输出, exp的求导结果与输出一致, 为epx(x)

#### (16) 取log

```python
    def log(self):
        datas = np.log(self.datas + 1e-5)

        def grad_fn1(grad):
            return grad / (self.datas + 1e-5)

        Tensors = []
        grad_fns = []
        Tensors.append(self)
        grad_fns.append(grad_fn1)
        return addEdges(datas, Tensors, grad_fns)
```

调用numpy实现的取logf函数, 计算得到输出, log的求导结果为1/datas, 需要注意的是因为datas可能很接近0, 从而导致log(datas)和1/datas趋近于无穷导致无法继续训练, 所以这里增加了1e-5进行纠正

#### (17) 取softmax

```python
    def __softmax__(self):
        return softmax__(self)
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
```

在前向计算结果的时候为了防止e^x过大, 所以采用了每个数值减去全局的最大值.

在求导方面:

<img src="file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml15768\wps3.jpg" alt="img" style="zoom:67%;" />

 

<img src="file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml15768\wps4.jpg" alt="img" style="zoom: 67%;" />

 

所以我们只需要遍历每一个样本, 然后利用-np.dot(value[i:i+1,:].T,value[i:i+1,:]), 计算出每对的-aiaj,最后利用np.identity(size) * value[i:i+1,:] 对i=j的元素增加aj即可

另外值得注意的是, 对于三个维度的矩阵而言使用np.matmul相当于遍历了每一个样本的梯度进行梯度的传递

#### (18) 求sum

```python
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
```

如果是对整个数组求和, 则对于每个位置的变量求导都是1, 所以需要乘上一个与输入数据相同形状的全一矩阵, 如果是对于某一个维度求和的话, 需要先在grad上添加对应的维度, 然后乘上一个和输入数据相同形状的全一矩阵

#### (19) im2col介绍及实现

##### a. **卷积**

卷积就是卷积核跟图像矩阵的运算. 卷积核是一个小窗口, 记录的是权重. 卷积核在输入图像上按步长滑动, 每次操作卷积核对应区域的输入图像, 将卷积核中的权值和对应的输入图像的值相乘再相加, 赋给卷积核中心所对应的输出特征图的一个值, 如下图所示.

<img src="file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml15768\wps5.jpg" alt="img" style="zoom:67%;" />

 

 

##### b. Im2col

在卷积运算中, 常常将图像和卷积核转化成矩阵, 从而将卷积运算转化成矩阵乘法运算, 进而在矩阵乘法的基础上进行计算优化, 而im2col就是将输入图像和卷积核转化成可以表示卷积操作的矩阵形式, 具体过程如下.

我们假设卷积核的尺寸为2*2，输入为3通道的3*3大小的图像. im2col做的事情就是对于卷积核每一次要处理的小窗，将其展开到新矩阵的一行中的一部分，新矩阵的行数，就是对于一副输入图像，卷积运算的次数（卷积核滑动的次数），

如下图所示, 其展开的矩阵大小为4*16, 形式地表示行数为outRow*outCol, 列数为chnnals*kernelRow*kernelCol, 而卷积核类似, 相当于用一个与卷积核大小相同的窗口对卷积核进行卷积, 在这种情况下展开的话, 展开后的矩阵行数为1, 列数为chnnals*kernelRow*kernleCol, 并且可以发现这样的两个矩阵相乘实际上就是做了一次卷积, 当然, 如果是batchSize不为1的话, 上面的矩阵形状可能会有一些细微的差别.

<img src="file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml15768\wps6.jpg" alt="img" style="zoom:67%;" />

 

 

##### c. **代码实现**

```python
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
```

首先获取数据各个维度的信息, 后通过padding参数进行0填充, 根据stride和padding计算得到输出图像的大小, 最后根据b部分的介绍, 将其通过循环转化成我们需要的形式.

#### (20) col2img实现

```python
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
```

与img2col类似, 实现一遍逆操作即可, 将转化后的矩阵恢复成原始的图像

#### (21) **conv_****实现**

```python
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
        imgTmp = im2col(img, kernelRow, kernelCol, stride, paddingRow, paddingCol).reshape(batchSize, outRow * outCol, channals, kernelRow * kernelCol).transpose(2,3,1,0).reshape(kernelRow * kernelCol * channals,outRow * outCol * batchSize).T
        grad_w = (grad.transpose(1, 2, 3, 0).reshape(out.shape[1], -1) @ imgTmp).reshape(kernel.shape)
        return grad_w

    def grad_bias(grad):
        grad = grad.reshape(out.shape)
        grad = np.sum(grad, axis=(0, 2, 3), keepdims=True)
        grad = grad.reshape(grad.shape[0], grad.shape[1])
        return grad


    def grad_func_img(grad):
        grad = grad.reshape(out.shape)
        flip_kernel = np.flipud(np.fliplr(np.flip(kernel))).transpose(1,0,2,3)
        if stride > 1:
            delta = np.zeros((out.shape[0],out.shape[1],outRow,outCol))
            delta[:,:,::stride,::stride] = grad
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
```

前向计算部分, 根据(19)b部分的介绍, 首先将img转化成(BatchSize * outRow * outCol, inputChannels * kernelRow * kernelCol)的矩阵, kernel转化成(outputChnnels, inputChannels * kernelRow * kernelCol)的矩阵, 通过上述形式, 可以直接通过矩阵乘法实现卷积，并且增加偏置项b, 得到的矩阵为(BatchSize * outRow * outCol, outputChnnels), 最后对形状进行调整得到(BatchSize, outputChnnels, outRow, outCol)的输出.

对于卷积核的求导, 我们首先将grad转化成(channels, batchSize * outRow * outCol)的形式, 将img转化成(kernelRow * kernelCol * channels, outRow * outCol*BatchSize)的形式, 结果梯度为grad与转化后的img的乘积, 最后记得将梯度转化成kernel的形状. 其实本过程很像前向计算的一个逆操作.

对于偏置bias的求导, grad的初始化形状是(BatchSize, outputChnnels, outRow, outCol), bias的形状是(1,outputChnnels), 在表达式中对bias的求导其实都是1, 并且这里类似于前面对于numpy中广播机制的处理类似, 所以我们需要将grad中广播扩展的维度求和, 也就是(0, 2, 3)这三个维度.

对于img的求导, grad转化成(BatchSize * imgRow * imgCol,  outputChannels * kernelRow * kernelCol)的形式, kernel需要先flip下, 后通过img2col转化成(iutputChnnels, onputChannels * kernelRow * kernelCol)的矩阵, 通过乘法运算获得(BatchSize * imgRow * imgCol, iutputChnnels)的结果，最后再根据padding调整下大小, 将其还原成img的输入形状即可

#### (22) **maxpool_****实现**

 ```python

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
 ```

前向计算部分, 根据(19)b部分的介绍, 首先将img转化成(BatchSize * outRow * outCol, inputChannels * kernelRow * kernelCol)的矩阵, 不难发现, 现在对第二维求最大值, 结果即为最大值池化的结果, 最后将其转化成输出的形状, 并且需要记录下最大值的位置, 方面后续求导

对img求导, 在反向传播的时候将grad传播至原始图像的最大值位置即可, 如下图所示.

![img](file:///C:\Users\Administrator\AppData\Local\Temp\ksohtml15768\wps7.jpg)