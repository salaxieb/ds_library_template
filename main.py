import pickle
from pathlib import Path

import numpy as np
from datasets import load_dataset

from language_model import (
    DataSupplier,
    Embedding,
    FullyConnected,
    MultyHeadSelfAttention,
    Perplexity,
    PositionalEncoding,
    SelfAttentionHead,
    SoftmaxCrossEntropyLoss,
    Tokenizer,
)

if __name__ == "__main__":
    ##############
    ### config ###
    ##############
    vocab_size = 5000
    context_size = 200
    batch_size = 64
    embedding_size = 264
    transformer_head_embedding_size = 64
    num_heads = 12
    num_mh_self_attention_layers = 12
    learning_rate = 0.001

    ###############
    ### dataset ###
    ###############
    dataset_path = Path("dataset")
    dataset_path.mkdir(exist_ok=True)

    if not (dataset_path / "tiny_shakespeare.txt").exists():
        d = load_dataset("tiny_shakespeare")
        with open(dataset_path / "tiny_shakespeare.txt", "w") as f:
            for subset_name, subset in d.data.items():
                subset = subset.to_pandas()
                for line in subset.text:
                    f.write(str(line))

    #################
    ### tokenizer ###
    #################
    tokenizer_path = Path("trained_tokenizer_vocab")
    tokenizer_path.mkdir(exist_ok=True)
    vocab_path = tokenizer_path / "vocab.txt"

    if not vocab_path.exists():
        tokenizer = Tokenizer(vocab_size=vocab_size)
        tokenizer.fit(dataset_path, save_path=vocab_path)

    tokenizer = Tokenizer().from_pretrained(vocab_path)
    test_msg = "example message for encoder and decoder"
    encoded = tokenizer.encode(test_msg)
    assert tokenizer.decode(encoded) == test_msg

    ################
    ### data gen ###
    ################
    data_supplier = DataSupplier(
        tokenizer,
        corpus_path=dataset_path,
        context_size=context_size,
        batch_size=batch_size,
    )

    metrics = {"perplexity": Perplexity()}

    #############
    ### model ###
    #############
    save_name = Path("trained_model/trained_model.pkl")
    model = (
        [Embedding(vocab_size=vocab_size, embedding_size=embedding_size)]
        + [PositionalEncoding(context_size=context_size, embedding_size=embedding_size)]
        + [
            MultyHeadSelfAttention(
                input_embedding_size=embedding_size,
                context_size=context_size,
                output_embedding_size=embedding_size,
                internal_embedding_size=transformer_head_embedding_size,
                num_heads=num_heads,
            )
            for _ in range(num_mh_self_attention_layers)
        ]
        + [FullyConnected(input_neurons=embedding_size, output_neurons=vocab_size)]
    )
    if save_name.exists():
        with save_name.open("rb") as f:
            model = pickle.load(f)

    print("model initialised nb of params:", sum(layer.nb_of_params for layer in model))
    ###########
    ### fit ###
    ###########
    ce_loss = SoftmaxCrossEntropyLoss()
    for i, (input_batch, targets_batch) in enumerate(data_supplier):
        x = input_batch
        # forward
        for layer in model:
            x = layer(x)
        print(f"batch {i}", end=" ")
        print(f"loss {ce_loss.loss(targets_batch, x)}", end=" ")
        print(
            f"metrics:",
            [f"{name}: {func(targets_batch, x)}" for name, func in metrics.items()],
        )
        dE_dx = ce_loss.backward()
        # backward
        for layer in reversed(model):
            dE_dx = layer.backward(dE_dx)
        if i == 10:
            save_name.parent.mkdir(exist_ok=True)
            with save_name.open("wb") as f:
                model = pickle.dump(model, f)
                break

    #############
    ### infer ###
    #############
    # inp = input()
    inp = "MENENIUS:\n"
    print("input:", inp)
    tokens = tokenizer.encode(inp)

    for i in range(15):
        x = [tokens]
        for layer in model:
            x = layer(x)
        x = x[0, -1, :]
        x = np.exp(x) / sum(np.exp(x))
        token_id = np.random.choice(np.arange(vocab_size), p=x)
        if token_id == 0:
            break
        tokens.append(token_id)
        print("output:", tokenizer.decode(tokens), end="\r")
