from abc import ABC, abstractmethod

import numpy as np


class Layer:
    @abstractmethod
    def __init__(self):
        return

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, x: np.array) -> np.array:
        return

    @abstractmethod
    def backward(self, dE_dO):
        return


class Softmax(Layer):
    def __init__(self):
        self.out = np.zeros(1)
        return

    def forward(self, x):
        self.out = np.exp(x) / np.sum(np.exp(x))
        return self.out

    def backward(self, dE_dO):
        dE_dx = dE_dO * np.sum(np.outer(self.out, (1 - self.out)), axis=-1)
        print("dE_dx", dE_dx)
        return dE_dx


class FullyConnected(Layer):
    def __init__(self, input_neurons: int, output_neurons: int):
        self.b = np.random.randn(output_neurons)
        self.w = np.random.randn(input_neurons, output_neurons)

    def forward(
        self, x: "np.array shaped[input_neurons]"
    ) -> "np.array shaped[output_neurons]":
        self.x = x
        return x @ self.w + self.b

    def backward(self, dE_dO, learning_rate=0.01):
        dE_dx = np.sum(self.w @ dE_dO, axis=-1)
        dO_dw = self.x
        self.w -= learning_rate * np.outer(dE_dO, dO_dw).T
        dO_db = 1
        self.b -= learning_rate * (dE_dO * dO_db)
        return dE_dx


class Embedding(Layer):
    def __init__(self, vocab_size: int, embedding_size: int = 64):
        self.embeddings = np.random.randn(vocab_size, embedding_size)

    def forward(self, x):
        self.x = x
        return self.embeddings[x]

    def backward(self, dE_dO, learning_rate=0.01):
        self.embeddings[self.x] -= learning_rate * dE_dO * 1  # dO_dx
        return self.x  # dE_dx
