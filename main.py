from datasets import load_dataset
import numpy as np
from pathlib import Path

from language_model import Tokenizer
from language_model import (
    FullyConnected,
    Embedding,
    SoftmaxCrossEntropyLoss,
    MultyHeadSelfAttention,
)


if __name__ == "__main__":

    # ## dataset
    # dataset_path = Path("dataset")
    # dataset_path.mkdir(exist_ok=True)

    # if not (dataset_path / "tiny_shakespeare.txt").exists():
    #     d = load_dataset("tiny_shakespeare")
    #     with open(dataset_path / "tiny_shakespeare.txt", "w") as f:
    #         for subset_name, subset in d.data.items():
    #             subset = subset.to_pandas()
    #             for line in subset.text:
    #                 f.write(str(line))
    #                 f.write("\n")

    # ## tokenizer
    # tokenizer_path = Path("trained_tokenizer_vocab")
    # tokenizer_path.mkdir(exist_ok=True)
    # vocab_path = tokenizer_path / "vocab.txt"

    # if not vocab_path.exists():
    #     tokenizer = Tokenizer(vocab_size=2000)
    #     tokenizer.fit(dataset_path, save_path=vocab_path)

    # tokenizer = Tokenizer().from_pretrained(vocab_path)
    # encoded = tokenizer.encode(
    #     "First, you know Caius Marcius is chief enemy to the people.!"
    # )

    # print(encoded)
    # print(tokenizer.decode(encoded))

    vocab_size = 87
    context_size = 7
    batch_size = 3
    transformer_embedding_size = 64
    token_embedding_size = 24

    encoded = np.array(
        [[6, 2, 7, 8, 1, 9], [34, 2, 67, 78, 2, 5], [6, 2, 67, 86, 1, 8]]
    )
    inp = np.hstack((np.zeros((batch_size, 1)), encoded)).astype(int)
    target = np.hstack((encoded, np.zeros((batch_size, 1)))).astype(int)
    # target = np.array([np.zeros((context_size, vocab_size))[range(len(enc)), enc[1:] + [0]] for enc in  encoded])
    # print(encoded.reshape(-1, 1))
    # target = np.zeros((batch_size, context_size, vocab_size))
    # target[range(batch_size), range(context_size), encoded.reshape(-1)] = 1
    # print()
    # print(target)
    losses = []

    ## model
    emb = Embedding(vocab_size=vocab_size, embedding_size=token_embedding_size)
    self_att = MultyHeadSelfAttention(
        input_embedding_size=token_embedding_size,
        context_size=context_size,
        output_embedding_size=transformer_embedding_size,
        internal_embedding_size=48,
        num_heads=4,
    )
    # self_att2 = MultyHeadSelfAttention(
    #     input_embedding_size=token_embedding_size,
    #     context_size=context_size,
    #     output_embedding_size=transformer_embedding_size,
    #     internal_embedding_size=48,
    #     num_heads=4,
    # )
    connected = FullyConnected(
        input_neurons=transformer_embedding_size, output_neurons=vocab_size
    )
    ce_loss = SoftmaxCrossEntropyLoss(vocab_size=vocab_size)
    for i in range(50):
        embedding = emb(inp)
        # print('x emb', x.shape)
        x = self_att(embedding, embedding)
        # x = self_att2(x, embedding)
        # print('x att', x.shape)
        x = connected(x)
        # print('x conn', x.shape)
        loss = ce_loss.loss(target, x)
        losses.append(loss)
        # print("loss", loss)
        dE_dO = ce_loss.backward()
        # print('err act', dE_dO.shape)
        dE_dO = connected.backward(dE_dO)
        # print('err conn', dE_dO.shape)
        # dE_dO, dE_dEmb = self_att2.backward(dE_dO)
        dE_dO, dE_dEmb = self_att.backward(dE_dO)
        dE_dO = dE_dO + dE_dEmb
        # print('err self att', dE_dO.shape)
        end = emb.backward(dE_dO)
        # print('end', end)

    print(losses)
