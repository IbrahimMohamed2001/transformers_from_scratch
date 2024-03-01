import torch
# import torch.nn as nn
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(self, dataset, source_tokenizer, target_tokenizer, source_language, target_language, max_len):
        super().__init__()

        self.dataset = dataset
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.source_language = source_language
        self.target_language = target_language
        self.max_len = max_len

        self.sos_token = torch.Tensor([source_tokenizer.token_to_id(['SOS'])], dtype=torch.int64)
        self.eos_token = torch.Tensor([source_tokenizer.token_to_id(['EOS'])], dtype=torch.int64)
        self.unk_token = torch.Tensor([source_tokenizer.token_to_id(['UNK'])], dtype=torch.int64)
        self.pad_token = torch.Tensor([source_tokenizer.token_to_id(['PAD'])], dtype=torch.int64)
        
        self.casual_mask = torch.tril(torch.ones(1, max_len, max_len)).int()

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        source_target_pair = self.dataset[index]
        source_text = source_target_pair['translation'][self.source_language]
        target_text = source_target_pair['translation'][self.target_language]

        enc_input_tokens = self.source_tokenizer.encode(source_text).ids
        dec_input_tokens = self.target_tokenizer.encode(target_text).ids

        enc_num_padding_tokens = self.max_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.max_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')

        encoder_input = torch.cat([
            self.sos_token, 
            torch.tensor(enc_input_tokens, dtype=torch.int64), 
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ])

        decoder_input = torch.cat([
            self.sos_token, 
            torch.tensor(dec_input_tokens, dtype=torch.int64), 
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ])

        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64), 
            self.eos_token, 
            torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ])

        return {
            'encoder_input': encoder_input, #! (max_len)
            'decoder_input': decoder_input, #! (max_len)
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), #! (1, 1, max_len)
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & self.casual_mask, #! (1, 1, max_len) & (1, max_len, max_len) -> (1, max_len, max_len)
            'label': label, #! (max_len)
            'source_text': source_text,
            'target_text': target_text
        }

    @staticmethod
    def casual_mask(size):
        return torch.tril(torch.ones(1, size, size)).int()