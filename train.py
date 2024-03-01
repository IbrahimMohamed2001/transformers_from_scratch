from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, random_split
# import torch.nn as nn
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from dataset import TranslationDataset
# import re
# import tqdm


def get_all_sentences(dataset, language):
    for item in dataset:
        yield item['translation'][language]


def get_or_build_tokenizer(config, dataset, language):
    tokenizer_path = Path(config['tokenizer_file'].format(language))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='UNK'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['UNK', 'PAD', 'SOS', 'EOS'], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

def get_dataset(name, config):
    dataset = load_dataset(name, f'{config['source_language']}-{config['target_language']}', split='train')

    source_tokenizer = get_or_build_tokenizer(config, dataset, config['source_language'])
    target_tokenizer = get_or_build_tokenizer(config, dataset, config['target_language'])

    train_size = len(0.9 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    train_dataset