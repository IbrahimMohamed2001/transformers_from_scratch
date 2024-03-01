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
from tqdm import tqdm
# import re


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

    train_dataset = TranslationDataset(train_dataset, source_tokenizer, target_tokenizer, config['source_language'], config['target_language'], config['max_len'])
    valid_dataset = TranslationDataset(valid_dataset, source_tokenizer, target_tokenizer, config['source_language'], config['target_language'], config['max_len'])

    # max_source_len = 0
    # max_target_len = 0

    # for item in tqdm(dataset):
    #     source_ids = source_tokenizer.encode(item['translation'][config['source_language']])
    #     max_source_len = max(max_source_len, len(source_ids))
        
    #     target_ids = target_tokenizer.encode(item['translation'][config['target_language']])
    #     max_target_len = max(max_target_len, len(target_ids))

    # print(f'the maximum source length is {max_source_len}')
    # print(f'the maximum target length is {max_target_len}')

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

    return {
        'train_dataloader': train_loader,
        'valid_dataloader': valid_loader,
        'source_tokenizer': source_tokenizer,
        'target_tokenizer': target_tokenizer
    }