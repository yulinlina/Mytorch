
import  numpy as np


def accuracy(a, y):
    size = a.shape[0]
    idx_a = np.argmax(a, axis=1)
    idx_y = np.argmax(y, axis=1)
    # cp = ClassificationPerformance(idx_a,idx_y)
    # return cp.getAccuracy()
    acc = sum(idx_a == idx_y)
    return acc