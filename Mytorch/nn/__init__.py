from Mytorch.nn import linear,conv2d,module,sequential
from Mytorch.nn import activation_function

def Linear():
    return linear.Linear

def CrossEntropy():
    return activation_function.CrossEntropyLoss

def Module():
    return module.Module


def Sequential():
    return sequential.Sequential