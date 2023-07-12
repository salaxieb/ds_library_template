import numpy as np


class SoftmaxCrossEntropyLoss:
    def __init__(self):
        return

    def loss(self, target, x):
        self.target = (np.arange(x.shape[-1]) == target[..., None]).astype(int)
        # if target is padding, don't take it to account
        self.probabilities = np.exp(x) / (
            np.sum(np.exp(x), axis=-1)[:, :, np.newaxis] + 1e-10
        )
        if np.any(np.isnan(self.target * np.log(self.probabilities))):
            print("probs.shape", self.probabilities.shape)
        return np.mean(-np.sum(self.target * np.log(self.probabilities), axis=-1))

    def backward(self):
        dE_dy = self.probabilities - self.target
        # set padding prediction loss to 0
        dE_dy[self.target[:, :, 0] == 1] = 0
        return dE_dy
