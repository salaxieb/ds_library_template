from pathlib import Path

import numpy as np

from language_model.tokenizer.tokenizer import Tokenizer


class DataSupplier:
    def __init__(
        self,
        tokenizer: Tokenizer,
        corpus_path: Path,
        context_size: int,
        batch_size: int,
    ):
        self.tokenizer = tokenizer
        self.datasets = list(corpus_path.glob("*.txt"))
        self.context_size = context_size
        self.batch_size = batch_size

    def __iter__(self):
        for dataset_file in self.datasets:
            with open(dataset_file) as f:
                texts = f.read().split("\n\n")
                for text in texts:
                    whole_encoded = self.tokenizer.encode(text)
                    encoded_batch = None
                    for ending_index in range(1, len(whole_encoded) + 1):
                        padding = [self.tokenizer.token2id["<EOS>"]] * max(
                            0, self.context_size - ending_index + 1
                        )
                        starting_index = 0
                        if ending_index > self.context_size:
                            starting_index = ending_index - self.context_size - 1
                        encoded_batch = (
                            np.vstack(
                                (
                                    encoded_batch,
                                    np.array(
                                        [
                                            padding
                                            + whole_encoded[starting_index:ending_index]
                                        ]
                                    ),
                                )
                            )
                            if encoded_batch is not None
                            else np.array(
                                [padding + whole_encoded[starting_index:ending_index]]
                            )
                        )

                        if encoded_batch.shape[0] == self.batch_size:
                            inputs = encoded_batch[:, :-1]
                            targets = encoded_batch[:, 1:]
                            encoded_batch = None
                            yield inputs, targets
