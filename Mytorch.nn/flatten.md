---
sort: 4
---
# Flatten

CLASS Mytorch.nn.Flatten(module_name,inputs)
将tensor拉成一维

---

参数：

- moudel_name
- inputs:Tensor

---

shape:(batch_size, -1)
```python

import numpy as np
from .module import Module
from ..tensor import *


class Flatten(Module):
    def __init__(self, module_name: str):
        super(Flatten, self).__init__(module_name)

    def forward(self, inputs: Tensor):
        assert isinstance(inputs, Tensor)
        batch_size = inputs.datas.shape[0]
        inputs = Tensor.reshape(inputs, (batch_size, -1))
        return inputs


```
