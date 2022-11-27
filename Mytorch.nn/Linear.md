---
sort: 1
---


CLASS Mytorch.nn.Linear(module_name: str, num_in, num_out)

实现线性层

---


参数：

- module_name
- num_in
- num_out


---


Shape:


```python
mport numpy as np
from .module import Module
from .parameter import Parameter
from ..tensor import *


class Linear(Module):

    def __init__(self, module_name: str, num_in, num_out):
        super(Linear, self).__init__(module_name)
        self.num_in = num_in
        self.num_out = num_out
        self.params.w_dict[module_name] = Tensor(np.zeros((num_in, num_out)), requires_grad=True)
        self.params.b_dict[module_name] = Tensor(np.zeros((1, num_out)), requires_grad=True)
        self.reset_parameters()

        # self.params.w_grad_dict[module_name] = Tensor(np.zeros_like(self.params.w_dict[module_name]), requires_grad=True)
        # self.params.b_grad_dict[module_name] = Tensor(np.zeros_like(self.params.b_dict[module_name]), requires_grad=True)

        # self.params.z_dict[module_name] = None
        # self.params.a_dict[module_name] = None
        # self.params.delta_dict[module_name] = None

    def reset_parameters(self):
        bound = np.sqrt(6. / (self.num_in + self.num_out))
        self.params.w_dict[self.module_name] = Tensor(np.random.uniform(-bound, bound, (self.num_in, self.num_out)), requires_grad=True)
        self.params.b_dict[self.module_name] = Tensor(np.random.uniform(-bound, bound, (1, self.num_out)), requires_grad=True)


    def __call__(self, inputs):
        return self.forward(inputs)

    def forward(self, inputs):
        """
        前向计算
        w [num_in, num_out]
        b [1, num_out]
        :param inputs: 输入 [batch,in_num]
        :return: 输出 [batch,out_num]
        """
        w = self.params.w_dict[self.module_name]
        b = self.params.b_dict[self.module_name]
        inputs = inputs.reshape((inputs.shape[0], -1))
        z = Tensor.__matmul__(inputs, w) + b

        # z = np.dot(input, w) + b
        # self.params.z_dict[self.module_name] = z
        # self.params.a_dict[self.module_name] = z    # 如果没有使用激活函数，那么输出a等于净输入z
        return z

```
