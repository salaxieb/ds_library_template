import numpy as np


class SoftmaxCrossEntropyLoss:
    def __init__(self):
        return

    def loss(self, target, x):
        self.target = target
        self.probabilities = np.exp(x) / np.sum(np.exp(x))
        return -np.sum(target * np.log(self.probabilities))

    def err_diff(self):
        return self.probabilities - self.target
