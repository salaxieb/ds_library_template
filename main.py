from datasets import load_dataset
import numpy as np
from pathlib import Path

from language_model import Tokenizer
from language_model import FullyConnected, Embedding, SoftmaxCrossEntropyLoss


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

    encoded = [
        67,
        1753,
        90,
        24,
        126,
        1755,
        35,
        42,
        627,
        101,
        42,
        75,
        77,
        627,
        1756,
        110,
        17,
        918,
        59,
        32,
        6,
        70,
        12,
        795,
        1757,
        13,
        6,
        16,
        19,
        133,
        8,
        111,
    ]
    encoded = [67] * 20

    ## model
    emb = Embedding(vocab_size=2000, embedding_size=64)
    connected = FullyConnected(input_neurons=64, output_neurons=5)
    ce_loss = SoftmaxCrossEntropyLoss()

    target = np.array([0, 0, 1, 0, 0])
    for enc in encoded:
        # print('enc', enc)
        x = emb(enc)
        # print('x emb', x)
        x = connected(x)
        # print('x conn', x)
        loss = ce_loss.loss(target, x)
        print("loss", loss)
        dE_dO = ce_loss.err_diff()
        # print('err act', dE_dO)
        dE_dO = connected.backward(dE_dO)
        # print('err conn', dE_dO)
        end = emb.backward(dE_dO)
        # print('end', end)
