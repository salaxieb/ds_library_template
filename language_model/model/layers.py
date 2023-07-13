from abc import ABC, abstractmethod

import numpy as np


class Layer:
    @abstractmethod
    def __init__(self):
        return

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    @abstractmethod
    def backward(self, dE_dO: np.ndarray) -> np.ndarray:
        return dE_dO

    @property
    def nb_of_params(self) -> int:
        return 0


class Softmax(Layer):
    def __init__(self, axis=-1):
        self.out = np.zeros(1)
        self.axis = axis
        return

    def forward(self, x):
        self.out = np.exp(x) / (np.sum(np.exp(x), axis=self.axis)[:, :, np.newaxis])
        return self.out

    def backward(self, dE_dO):
        dE_dx = (
            self.out
            * (
                np.tile([np.eye(self.out.shape[1])], (self.out.shape[0], 1, 1))
                - self.out
            )
        ) * dE_dO
        return dE_dx


class Elu(Layer):
    def __init__(self, alpha = 0.2):
        self.alpha = alpha

    def forward(self, x: np.ndarray) -> np.ndarray:
        x[x < 0] = np.exp(x[x<0]) - 1
        self.out = x
        return self.out

    def backward(self, dE_dO: np.ndarray) -> np.ndarray:
        self.out[self.out > 0] = 0
        return self.out + 1


class FullyConnected(Layer):
    def __init__(
        self, input_neurons: int, output_neurons: int, norm_constant: int = 10
    ):
        self.b = np.random.randn(output_neurons) / norm_constant
        self.w = np.random.randn(input_neurons, output_neurons) / norm_constant

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return x @ self.w + self.b

    def backward(self, dE_dO, learning_rate=0.001):
        dE_dx = dE_dO @ self.w.T
        dO_dw = self.x
        self.w -= learning_rate * np.tensordot(dO_dw, dE_dO, axes=((0, 1), (0, 1)))
        dO_db = 1
        self.b -= learning_rate * np.sum(dE_dO, axis=(0, 1)) * dO_db
        return dE_dx

    @property
    def nb_of_params(self):
        return self.w.size + self.b.size


class Embedding(Layer):
    def __init__(
        self, vocab_size: int, embedding_size: int = 64, norm_constant: int = 10
    ):
        self.embeddings = np.random.randn(vocab_size, embedding_size) / norm_constant

    def forward(self, x):
        self.x = x
        return self.embeddings[x]

    def backward(self, dE_dO, learning_rate=0.001):
        self.embeddings[self.x] -= learning_rate * dE_dO * 1  # dO_dx
        return self.x  # dE_dx

    @property
    def nb_of_params(self):
        return self.embeddings.size


class PositionalEncoding(Layer):
    def __init__(self, context_size: int, embedding_size: int):
        longitude = np.repeat([np.arange(context_size) / 100], embedding_size, axis=0).T
        latitude = np.repeat([np.arange(embedding_size) / 100], context_size, axis=0)
        encoded_position = latitude * longitude
        self.pe = np.zeros((1, context_size, embedding_size))
        self.pe[:, 0::2, :] = np.sin(encoded_position[0::2, :])
        self.pe[:, 1::2, :] = np.cos(encoded_position[1::2, :])

    def forward(self, x: np.ndarray):
        return x + np.repeat(self.pe, x.shape[0], axis=0)

    def backward(self, dE_dx: np.ndarray):
        return dE_dx


class FeedForwardDotProduct(Layer):
    def __init__(self, embedding_size: int, output_size: int, norm_constant: int = 10):
        self.W = np.random.randn(embedding_size, output_size) / norm_constant

    def forward(self, x: np.ndarray):
        self.x = x  # shape context_size x embedding_size
        return self.x @ self.W

    def backward(self, dE_dO, learning_rate=0.001):
        # dE_dO shape context_size x output_size
        dE_dx = dE_dO @ self.W.T  # context_size X embedding_size
        dO_dW = self.x  # embedding_size x output_size
        self.W -= learning_rate * np.tensordot(dO_dW, dE_dO, axes=((0, 1), (0, 1)))
        return dE_dx

    @property
    def nb_of_params(self):
        return self.W.size


class TensorsDotProduct(Layer):
    def __init__(self):
        return

    def forward(self, x1: np.ndarray, x2: np.ndarray):  # type: ignore
        self.x1 = x1  # bs x a x b
        self.x2 = x2  # bs x b x c
        return np.einsum("ijk,ikp->ijp", x1, x2)  # bs x a x c

    def backward(self, dE_dO):
        # dE_dO a x c
        dE_dx1 = np.einsum("ijp,ikp->ijk", dE_dO, self.x2)  # bs x a x b
        dE_dx2 = np.einsum("ijk,ijp->ikp", self.x1, dE_dO)  # bs x a x c
        return dE_dx1, dE_dx2


class Transpose(Layer):
    def __init__(self):
        return

    def forward(self, x):
        return np.transpose(x, (0, 2, 1))

    def backward(self, dE_dO):
        return np.transpose(dE_dO, (0, 2, 1))


class MultiplyConstant(Layer):
    def __init__(self, multiplier: float):
        self.multiplier = multiplier
        return

    def forward(self, x):
        return x * self.multiplier

    def backward(self, dE_dO):
        return dE_dO * self.multiplier


class Splitter(Layer):
    def __init__(self, split_to: int):
        self.split_to = split_to
        return

    def forward(self, x):
        return np.split(x, self.split_to, axis=-1)

    def backward(self, dE_dO_arr):
        return np.concatenate(dE_dO_arr, axis=-1)


class Concatenator(Layer):
    def __init__(self, concatenate_from: int):
        self.concatenate_from = concatenate_from
        return

    def forward(self, x):
        return np.concatenate(x, axis=-1)

    def backward(self, dE_dO_arr):
        return np.split(dE_dO_arr, self.concatenate_from, axis=-1)


class LayerNormalisation(Layer):
    def __init__(self, dimensions: int):
        self.dimensions = dimensions
        return

    def forward(self, features: np.ndarray):
        mean = np.mean(features, axis=tuple(range(1, self.dimensions)))
        mean = np.kron(
            mean.reshape(-1, *(1,) * (self.dimensions - 1)),
            np.ones((1, *features.shape[1:])),
        )
        self.std = np.std(features, axis=tuple(range(1, self.dimensions)))
        self.std = np.kron(
            self.std.reshape(-1, *(1,) * (self.dimensions - 1)),
            np.ones((1, *features.shape[1:])),
        )
        # this step is required, because limitations of np
        # layer normilisation returns large numbers which raise errors in gradient
        self.std = 10 * self.std
        return (features - mean) / (self.std + 1e-10)

    def backward(self, dE_dO):
        return dE_dO / (self.std + 1e-10)


class Add(Layer):
    def __init__(self):
        return

    def forward(self, x1, x2):
        return x1 + x2

    def backward(self, dE_dO):
        return dE_dO, np.copy(dE_dO)


class SelfAttentionHead(Layer):
    def __init__(
        self, input_embedding_size: int, context_size: int, q_k_v_size: int = 32
    ):
        self.q_k_v_size = q_k_v_size
        self.Wq_layer = FeedForwardDotProduct(input_embedding_size, q_k_v_size)
        self.Wk_layer = FeedForwardDotProduct(input_embedding_size, q_k_v_size)
        self.Wv_layer = FeedForwardDotProduct(input_embedding_size, q_k_v_size)
        self.transpose = Transpose()
        self.dot_prod_Q_K = TensorsDotProduct()
        self.dot_prod_wights_V = TensorsDotProduct()
        self.multiply = MultiplyConstant(1 / self.q_k_v_size**0.5)
        self.softmax = Softmax()

    def forward(self, x: np.ndarray):
        # x shaped [context_size x embedding_size]
        Q = self.Wq_layer(x)  # out shape [context_size x q_k_v_size]
        K = self.Wk_layer(x)  # out shape [context_size x q_k_v_size]
        V = self.Wv_layer(x)  # out shape [context_size x q_k_v_size]

        Q = self.transpose(Q)
        weights = self.dot_prod_Q_K(K, Q)
        weights = self.multiply(weights)
        triu_indices = np.triu_indices(n=weights.shape[1], k=1)
        weights[:, triu_indices[0], triu_indices[1]] = -np.inf
        weights = self.softmax(weights)
        out = self.dot_prod_wights_V(weights, V)  # out shape context_size x q_k_vsize
        return out

    def backward(self, dE_dO, learning_rate: float = 0.01):
        dE_dweights, dE_dV = self.dot_prod_wights_V.backward(dE_dO)
        dE_dweights = self.softmax.backward(dE_dweights)
        dE_dweights = self.multiply.backward(dE_dweights)
        dE_dK, dE_dQ = self.dot_prod_Q_K.backward(dE_dweights)
        dE_dQ = self.transpose.backward(dE_dQ)
        dE_dxq = self.Wq_layer.backward(dE_dQ, learning_rate=learning_rate)
        dE_dxk = self.Wk_layer.backward(dE_dK, learning_rate=learning_rate)
        dE_dxv = self.Wv_layer.backward(dE_dV, learning_rate=learning_rate)
        dE_dx = dE_dxq + dE_dxk + dE_dxv
        return dE_dx

    @property
    def nb_of_params(self):
        return (
            self.Wq_layer.nb_of_params
            + self.Wk_layer.nb_of_params
            + self.Wv_layer.nb_of_params
        )


class MultyHeadSelfAttention(Layer):
    def __init__(
        self,
        input_embedding_size: int,
        context_size: int,
        output_embedding_size: int,
        internal_embedding_size: int,
        num_heads: int,
    ):
        assert (
            input_embedding_size / num_heads
        ) % 1 == 0, "embedding size must be devisible by number of heads"
        self.heads = [
            SelfAttentionHead(
                input_embedding_size // num_heads, context_size, internal_embedding_size
            )
            for _ in range(num_heads)
        ]
        self.splitter = Splitter(split_to=num_heads)
        self.concat = Concatenator(concatenate_from=num_heads)
        self.dot_prod_Z_W = FeedForwardDotProduct(
            embedding_size=internal_embedding_size * num_heads,
            output_size=output_embedding_size,
        )
        self.add = Add()
        self.layer_norm_1 = LayerNormalisation(dimensions=3)
        self.feed_forward = FullyConnected(
            input_neurons=output_embedding_size, output_neurons=output_embedding_size
        )
        self.elu = Elu()
        self.layer_norm_2 = LayerNormalisation(dimensions=3)

    def forward(self, x: np.ndarray):
        x_input = x.copy()
        x_arr = self.splitter(x)
        x_arr = [head(x) for x, head in zip(x_arr, self.heads)]
        x = self.concat(x_arr)
        x = self.dot_prod_Z_W(x)
        x = self.add(x, x_input)
        norm_sum_x = self.layer_norm_1(x)
        x = self.feed_forward(norm_sum_x.copy())
        x = self.elu(x)
        x = self.add(x, norm_sum_x)
        # print('feed_forward', x.shape)
        x = self.layer_norm_2(x)
        return x

    def backward(self, dE_dx, learning_rate: float = 0.001):
        dE_dx = self.layer_norm_2.backward(dE_dx)
        dE_dx, dE_norm_sum_x = self.add.backward(dE_dx)
        dE_dx = self.elu.backward(dE_dx)
        dE_dx = self.feed_forward.backward(dE_dx, learning_rate=learning_rate)
        dE_dx = dE_dx + dE_norm_sum_x
        dE_dx = self.layer_norm_1.backward(dE_dx)
        dE_dx, dE_dx_input = self.add.backward(dE_dx)
        dE_dx = self.dot_prod_Z_W.backward(dE_dx, learning_rate=learning_rate)
        dE_dx_arr = self.concat.backward(dE_dx)
        dE_dx_arr = [
            head.backward(dE_dx_head, learning_rate=learning_rate)
            for dE_dx_head, head in zip(dE_dx_arr, self.heads)
        ]
        dE_dx = self.splitter.backward(dE_dx_arr)
        dE_dx = dE_dx + dE_dx_input
        return dE_dx

    @property
    def nb_of_params(self):
        return (
            sum(head.nb_of_params for head in self.heads)
            + self.dot_prod_Z_W.nb_of_params
            + self.feed_forward.nb_of_params
        )
