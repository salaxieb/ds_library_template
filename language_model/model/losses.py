import numpy as np


class SoftmaxCrossEntropyLoss:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        return

    def loss(self, target, x):
        self.target = (np.arange(self.vocab_size) == target[..., None] - 1).astype(int)
        # self.target = target
        # self.probabilities = np.exp(x) / np.sum(np.exp(x))
        self.probabilities = np.exp(x) / np.sum(np.exp(x), axis=-1)[:, :, np.newaxis]
        print("probs.shape", self.probabilities.shape)
        return np.mean(-np.sum(self.target * np.log(self.probabilities), axis=-1))

    def backward(self):
        return self.probabilities - self.target
