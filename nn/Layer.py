from nn.Module import  Module
class sequential(Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            self._modules[block] = block

    def forward(self, x):
        for block in self._modules.values():
            x = block(x)
        return x