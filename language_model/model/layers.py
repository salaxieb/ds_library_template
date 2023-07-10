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
    def __init__(self, axis=-1):
        self.out = np.zeros(1)
        self.axis = axis
        return

    def forward(self, x):
        # # print('x inp', x)
        # x = np.exp(x)
        # # print('exp x', np.exp(x))
        # # print(f'sum by axis {self.axis}', np.sum(np.exp(x), axis=self.axis))
        # # print(np.exp(x))
        # # print('sum in softmax', np.sum(np.exp(x), axis=self.axis)[:, np.newaxis])
        self.out = np.exp(x) / np.sum(np.exp(x), axis=self.axis)[:, :, np.newaxis]
        # print(' x out', self.out)
        return self.out

    def backward(self, dE_dO):
        # print('dE_dO', dE_dO.shape)
        # # print('dE_dO', dE_dO)
        # print('softmax out', self.out.shape)
        # # print('multipl', (self.out * (1 - self.out)))
        dE_dx = (self.out * (1 - self.out)) @ dE_dO
        # print("dE_dx", dE_dx.shape)
        return dE_dx


class FullyConnected(Layer):
    def __init__(self, input_neurons: int, output_neurons: int):
        self.b = np.random.randn(output_neurons) / 3
        self.w = np.random.randn(input_neurons, output_neurons) / 3

    def forward(
        self, x: "np.array shaped[input_neurons]"
    ) -> "np.array shaped[output_neurons]":
        self.x = x
        return x @ self.w + self.b

    def backward(self, dE_dO, learning_rate=0.01):
        # # print(self.w.shape)
        # # print(dE_dO.shape)
        dE_dx = dE_dO @ self.w.T
        # # print('dE_dx', dE_dx.shape)
        # # print(dE_dx.shape)
        dO_dw = self.x
        # # print(dE_dO.shape, dO_dw.shape)
        # # print(np.tensordot(dO_dw, dE_dO, axes=((0,1), (0,1))).shape)
        self.w -= learning_rate * np.tensordot(dO_dw, dE_dO, axes=((0, 1), (0, 1)))
        dO_db = 1
        # # print(dE_dO.shape)
        # # print((dO_dw.T @ dE_dO).shape)
        self.b -= learning_rate * np.sum(dE_dO, axis=(0, 1)) * dO_db
        return dE_dx


class Embedding(Layer):
    def __init__(self, vocab_size: int, embedding_size: int = 64):
        self.embeddings = np.random.randn(vocab_size, embedding_size) / 3

    def forward(self, x):
        self.x = x
        # print('x', x)
        return self.embeddings[x]

    def backward(self, dE_dO, learning_rate=0.01):
        print(dE_dO.shape)
        self.embeddings[self.x] -= learning_rate * dE_dO * 1  # dO_dx
        return self.x  # dE_dx


class DotProduct(Layer):
    def __init__(self, embedding_size: int, output_size: int, norm_constant: int = 3):
        self.W = np.random.randn(embedding_size, output_size) / norm_constant

    def forward(self, x: np.array):
        self.x = x  # shape context_size x embedding_size
        return self.x @ self.W

    def backward(self, dE_dO, learning_rate=0.01):
        # dE_dO shape context_size x output_size
        # print('self.x, dE_dO, self.W', self.x.shape, dE_dO.shape, self.W.shape)
        dE_dx = dE_dO @ self.W.T  # context_size X embedding_size
        # print('dE_dx', dE_dx.shape)
        dO_dW = self.x  # embedding_size x output_size
        self.W -= learning_rate * np.tensordot(dO_dW, dE_dO, axes=((0, 1), (0, 1)))
        return dE_dx


class TwoInputDotProduct(Layer):
    def __init__(self):
        return

    def forward(self, x1: np.array, x2: np.array):
        # print('x1 x2', x1.shape, x2.shape)
        self.x1 = x1  # bs x a x b
        self.x2 = x2  # bs x b x c
        return np.einsum("ijk,ikp->ijp", x1, x2)  # bs x a x c

    def backward(self, dE_dO):
        # dE_dO a x c
        # dE_dx1 a x b
        # dE_dx2 b x c
        # print('dE_dO, self.x2', dE_dO.shape, self.x2.shape)
        dE_dx1 = np.einsum("ijp,ikp->ijk", dE_dO, self.x2)  # bs x a x b
        # dE_dx1 = np.tensordot(dE_dO, self.x2, axes=(0))
        # print('dE_dx1', dE_dx1.shape)
        # print('self.x1, dE_dO', self.x1.shape, dE_dO.shape)
        # dE_dx2 = self.x1.T @ dE_dO
        dE_dx2 = np.einsum("ijk,ijp->ikp", self.x1, dE_dO)  # bs x a x c
        # print('dE_dx2', dE_dx2.shape)
        return dE_dx1, dE_dx2


class Transpose(Layer):
    def __init__(self):
        return

    def forward(self, x):
        return np.transpose(x, (0, 2, 1))

    def backward(self, dE_dO):
        return np.transpose(dE_dO, (0, 2, 1))


class Multiply(Layer):
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
        print("inp", x.shape)
        print("splitted", len(np.split(x, self.split_to, axis=-1)))
        return np.split(x, self.split_to, axis=-1)

    def backward(self, dE_dO_arr):
        return np.concatenate(dE_dO_arr, axis=-1)


class Concatenator(Layer):
    def __init__(self, concatenate_from: int):
        self.concatenate_from = concatenate_from
        return

    def forward(self, x):
        # print('int', x.shape)
        print("concatenated", np.concatenate(x, axis=-1).shape)
        return np.concatenate(x, axis=-1)

    def backward(self, dE_dO_arr):
        return np.split(dE_dO_arr, self.concatenate_from, axis=-1)


class SelfAttentionHead(Layer):
    def __init__(
        self, input_embedding_size: int, context_size: int, q_k_v_size: int = 32
    ):
        self.q_k_v_size = q_k_v_size
        self.Wq_layer = DotProduct(input_embedding_size, q_k_v_size)
        self.Wk_layer = DotProduct(input_embedding_size, q_k_v_size)
        self.Wv_layer = DotProduct(input_embedding_size, q_k_v_size)
        self.transpose = Transpose()
        self.dot_prod_Q_K = TwoInputDotProduct()
        self.dot_prod_wights_V = TwoInputDotProduct()
        self.multiply = Multiply(1 / self.q_k_v_size**0.5)
        self.softmax = Softmax()

    def forward(self, x: np.array):
        # x shaped context_size, embedding_size
        Q = self.Wq_layer(x)  # out shape context_size x q_k_v_size
        # print('Q', Q.shape)
        K = self.Wk_layer(x)  # out shape context_size x q_k_v_size
        V = self.Wv_layer(x)  # out shape context_size x q_k_v_size

        Q = self.transpose(Q)
        weights1 = self.dot_prod_Q_K(K, Q)
        weights2 = self.multiply(weights1)
        # # print('weights 2', weights2.shape)        # masking future from previous steps

        weights2[
            :, np.triu_indices(n=7, k=1)[0], np.triu_indices(n=7, k=1)[1]
        ] = -np.inf
        # weights2 = np.apply_along_axis(lambda single_weights: single_weights[np.triu_indices_from(single_weights, k=1)] = -np.inf,
        # print('weights 2', weights2.shape)
        weights3 = self.softmax(weights2)
        # print('weights 3', weights3.shape)
        # # print('sum 1', np.sum(weights3, axis=0))
        # # print('sum 2', np.sum(weights3, axis=1))
        # # print('sum 3', np.sum(weights3, axis=2))
        # # print('V', V)
        out = self.dot_prod_wights_V(weights3, V)  # out shape context_size x q_k_vsize
        # print('out', out.shape)
        # # print('out', out)
        return out

    def backward(self, dE_dO):
        dE_dweights3, dE_dV = self.dot_prod_wights_V.backward(dE_dO)
        # print('dE_dV, dE_dweights3',dE_dV.shape, dE_dweights3.shape)
        dE_dweights2 = self.softmax.backward(dE_dweights3)
        # print('dE_dweights2', dE_dweights2.shape)
        dE_dweights1 = self.multiply.backward(dE_dweights2)
        # print('dE_dweights1', dE_dweights1.shape)
        dE_dK, dE_dQ = self.dot_prod_Q_K.backward(dE_dweights1)
        # print('dE_dK, dE_dQ', dE_dK.shape, dE_dQ.shape)
        dE_dQ = self.transpose.backward(dE_dQ)
        # print('dE_dK, dE_dQ', dE_dK.shape, dE_dQ.shape)
        dE_dxq = self.Wq_layer.backward(dE_dQ)
        # print('dE_dxq', dE_dxq.shape)
        dE_dxk = self.Wk_layer.backward(dE_dK)
        # print('dE_dxk', dE_dxk.shape)
        dE_dxv = self.Wv_layer.backward(dE_dV)
        # # print('dE_dxv', dE_dxv.shape)
        dE_dx = dE_dxq + dE_dxk + dE_dxv
        return dE_dx


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
        self.dot_prod_Z_W = DotProduct(
            embedding_size=internal_embedding_size * num_heads,
            output_size=input_embedding_size,
        )
        self.feed_forward = FullyConnected(
            input_neurons=input_embedding_size, output_neurons=output_embedding_size
        )

    def forward(self, x: np.array, embedding: np.array):
        x_arr = self.splitter(x)
        x_arr = [head(x) for x, head in zip(x_arr, self.heads)]
        x = self.concat(x_arr)
        x = self.dot_prod_Z_W(x)
        x = x + embedding
        x = self.feed_forward(x)
        return x

    def backward(self, dE_dO):
        dE_dx = self.feed_forward.backward(dE_dO)
        dE_dEmbedding = dE_dx  # fixme
        dE_dx = dE_dx  # fixme
        dE_dx = self.dot_prod_Z_W.backward(dE_dx)
        dE_dx_arr = self.concat.backward(dE_dx)
        dE_dx_arr = [
            head.backward(dE_dx_head) for dE_dx_head, head in zip(dE_dx_arr, self.heads)
        ]
        dE_dx = self.splitter.backward(dE_dx_arr)
        return dE_dx, dE_dEmbedding
