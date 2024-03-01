from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from model import Seq2SeqTransformer

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from dataset import TranslationDataset

import warnings
from config import get_weights_file_path, get_config
from torch.utils.tensorboard import SummaryWriter


# import re


def get_all_sentences(dataset, language):
    for item in dataset:
        yield item['translation'][language]


def get_or_build_tokenizer(config, dataset, language):
    tokenizer_path = Path(config['tokenizer_file'].format(language))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='<UNK>'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['<UNK>', '<PAD>', '<SOS>', '<EOS>'], min_frequency=2)
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

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True)

    return {
        'train_dataloader': train_loader,
        'valid_dataloader': valid_loader,
        'source_tokenizer': source_tokenizer,
        'target_tokenizer': target_tokenizer
    }

def get_model(config, source_vocab_size, target_vocab_size, ):
    return Seq2SeqTransformer(
        source_vocab_size, 
        target_vocab_size, 
        config['max_len'], 
        config['max_len'], 
        config['d_model'],
        config['num_layers'],
        config['heads'],
        config['hidden_size_ff'],
        config['dropout']
    )

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using device {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_loader = get_dataset('iwslt2017', config)['train_dataloader']
    valid_loader = get_dataset('iwslt2017', config)['valid_dataloader']
    source_tokenizer = get_dataset('iwslt2017', config)['source_tokenizer']
    target_tokenizer = get_dataset('iwslt2017', config)['target_tokenizer']

    model = get_model(config, source_tokenizer.get_vocab_size(), target_tokenizer.get_vocab_size()).to(device)

    writer = SummaryWriter(config['experiment_name'])

    optimizer = Adam(model.parameters(), lr=config['learning_rate'], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'preloading model {model_filename}')
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    criterion = nn.CrossEntropyLoss(ignore_index=source_tokenizer.token_to_id('<PAD>'), label_smoothing=0.1).to(device)
    epochs = config['num_epochs']

    for epoch in range(initial_epoch, epochs):
        model.train()
        batch_iterator = tqdm(train_loader, desc=f'processing epoch {epoch:02d}')
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) #! (B, max_len)
            decoder_input = batch['decoder_input'].to(device) #! (B, max_len)
            encoder_mask = batch['encoder_mask'].to(device) #! (B, 1, 1, max_len)
            decoder_mask = batch['decoder_mask'].to(device) #! (B, 1, max_len, max_len)
            label = batch['label'].to(device) #! (B, max_len)

            encoder_output = model.encode(encoder_input, encoder_mask) #! (B, max_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) #! (B, max_len, d_model)
            proj_output = model.project(decoder_output) #! (B, max_len, target_vocab_size)

            loss = criterion(proj_output.view(-1, target_tokenizer.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({'loss': f'{loss.item():6.3f}'})

            writer.add_scaler('train_loss', loss.item, global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        model_filename = get_weights_file_path(config, epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }, model_filename)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)