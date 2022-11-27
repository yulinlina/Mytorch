---
sort: 2
---


CLASS Mytorch.nn.Conv2d(module_name, intputChannal, outputChnnal, kernel_size, stride=1, padding=0))  

实现卷积操作

---


参数：

- module_name
- intputChannal
- outputChnnal,
- kernel_size,
- stride=1
- padding=0


---


Shape: (output_channel,input_channel,kernel_size)


```python
import numpy as np
from .module import Module
from ..tensor import *


import numpy as np
from .module import Module
from ..tensor import *


class Conv2d(Module):
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

```
