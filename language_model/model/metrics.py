import numpy as np
import torch
from torcheval.metrics.text import Perplexity as TorchPerplexity


class Perplexity:
    @staticmethod
    def __call__(target: np.ndarray, x: np.ndarray):
        probabilities = np.exp(x) / (
            np.sum(np.exp(x), axis=-1)[:, :, np.newaxis] + 1e-10
        )
        probabilities = probabilities[
            np.arange(probabilities.shape[2]) == target[..., None]
        ]
        return np.exp(-np.mean(np.log(probabilities)))
