---
sort: 5
---

# Maxpool


CLASS Maxpool(_**module_name, size, stride=1**_**)**:

实现卷积操作

---

参数：

- module_name
- size
- stride


---


Shape:


```python
class Maxpool(Module):
    def __init__(self, module_name, size, stride=1):
        super(Maxpool, self).__init__(module_name)
        self.size = size  # maxpool框的尺寸
        self.stride = stride

    def __call__(self, inputs, mode=True):
        return self.forward(inputs, mode)

    def forward(self, inputs, mode=True):
        # self.inputs = inputs
        return Tensor.__maxpool__(inputs, self.size, self.stride)

```
