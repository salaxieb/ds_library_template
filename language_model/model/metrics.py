import numpy as np
<<<<<<< HEAD
=======

>>>>>>> cf2d41b (added metrics, positional encoding, layer norm, model, data supplier and many more)
import torch
from torcheval.metrics.text import Perplexity as TorchPerplexity


class Perplexity:
    @staticmethod
<<<<<<< HEAD
    def __call__(target: np.ndarray, x: np.ndarray):
=======
    def __call__(target: np.array, x: np.array):
>>>>>>> cf2d41b (added metrics, positional encoding, layer norm, model, data supplier and many more)
        probabilities = np.exp(x) / (
            np.sum(np.exp(x), axis=-1)[:, :, np.newaxis] + 1e-10
        )
        probabilities = probabilities[
            np.arange(probabilities.shape[2]) == target[..., None]
        ]
        return np.exp(-np.mean(np.log(probabilities)))
