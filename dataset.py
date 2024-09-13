import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    """
    A PyTorch Dataset for translation tasks.

    Args:
        dataset (datasets.Dataset): Hugging Face dataset object containing translation pairs.
        source_tokenizer: Tokenizer for the source language.
        target_tokenizer: Tokenizer for the target language.
        source_language (str): Source language identifier.
        target_language (str): Target language identifier.
        max_len (int): Maximum sequence length.
    """

    def __init__(
        self,
        dataset,
        source_tokenizer,
        target_tokenizer,
        source_language,
        target_language,
        max_len,
    ):
        super().__init__()

        self.dataset = dataset
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_language = source_language
        self.target_language = target_language
        self.max_len = max_len

        self.sos_token = torch.tensor(
            [source_tokenizer.token_to_id("<SOS>")], dtype=torch.int64
        )
        self.eos_token = torch.tensor(
            [source_tokenizer.token_to_id("<EOS>")], dtype=torch.int64
        )
        self.unk_token = torch.tensor(
            [source_tokenizer.token_to_id("<UNK>")], dtype=torch.int64
        )
        self.pad_token = torch.tensor(
            [source_tokenizer.token_to_id("<PAD>")], dtype=torch.int64
        )

        self.causal_mask = TranslationDataset.causal_mask(max_len)

    def __len__(self):
        """
        Returns the number of translation pairs in the dataset.

        Returns:
            int: Number of translation pairs.
        """

        return len(self.dataset)

    def __getitem__(self, index):
        """
        Retrieves a translation pair from the dataset.

        Args:
            index (int): Index of the translation pair in the dataset.

        Returns:
            dict: Dictionary containing encoder and decoder inputs, masks, labels, and source/target texts.
        """

        source_target_pair = self.dataset[index]
        source_text = source_target_pair["translation"][self.source_language]
        target_text = source_target_pair["translation"][self.target_language]

        enc_input_tokens = self.source_tokenizer.encode(source_text).ids
        dec_input_tokens = self.target_tokenizer.encode(target_text).ids

        enc_num_padding_tokens = self.max_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.max_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * enc_num_padding_tokens, dtype=torch.int64
                ),
            ]
        )

        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor(
                    [self.pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ]
        )

        return {
            "encoder_input": encoder_input,  #! (max_len)
            "decoder_input": decoder_input,  #! (max_len)
            "encoder_mask": (encoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),  #! (1, 1, max_len)
            "decoder_mask": (decoder_input != self.pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int()
            & self.causal_mask,  #! (1, 1, max_len) & (1, max_len, max_len) -> (1, max_len, max_len)
            "label": label,  #! (max_len)
            "source_text": source_text,
            "target_text": target_text,
        }

    @staticmethod
    def causal_mask(size):
        """
        Generates a causal mask tensor of the specified size.

        Args:
            size (int): Size of the casual mask.

        Returns:
            Tensor: Casual mask tensor.
        """

        return torch.tril(torch.ones(1, size, size)).int()
