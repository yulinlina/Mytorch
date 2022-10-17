class Module():
    def __init__(self, Sequential):
        self.Sequential = Sequential
        self.mode = True

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return self.Sequential.forward(x, self.mode)

    def backward(self, output_delta):
        self.Sequential.backward(output_delta)


    def add_layer(self, layer):
        self.Sequential.add_layer(layer)

    def train(self):
        self.mode = True

    def eval(self):
        self.mode = False