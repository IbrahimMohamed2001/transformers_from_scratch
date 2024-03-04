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


def get_all_sentences(dataset, language):
    """
    Extracts all sentences in a specified language from a dataset.

    Args:
        dataset (datasets.Dataset): Hugging Face dataset object containing translation pairs.
        language (str): Language identifier for the desired translations.

    Yields:
        str: Sentences in the specified language from the dataset.
    """

    for item in dataset:
        yield item['translation'][language]


def get_or_build_tokenizer(config, dataset, language):
    """
    Retrieves a pre-trained tokenizer for a specified language or builds a new one if not available.

    Args:
        config (dict): Configuration dictionary containing parameters.
        dataset (datasets.Dataset): Hugging Face dataset object containing translation pairs.
        language (str): Language identifier for the desired tokenizer.

    Returns:
        Tokenizer: Tokenizer object for the specified language.
    """

    tokenizer_path = Path(config['tokenizer_file'].format(language))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='<UNK>'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=['<UNK>', '<PAD>', '<SOS>', '<EOS>'], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, language), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(name, config):
    """
    Loads a translation dataset, preprocesses it, and prepares data loaders for training and validation.

    Args:
        name (str): Name of the translation dataset to load.
        config (dict): Configuration dictionary containing parameters.

    Returns:
        dict: A dictionary containing train and validation data loaders, source and target tokenizers.
    """

    dataset = load_dataset(name, f'{name}-{config["source_language"]}-{config["target_language"]}', split='train')

    train_size = int(0.9 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

    source_tokenizer = get_or_build_tokenizer(config, dataset, config['source_language'])
    target_tokenizer = get_or_build_tokenizer(config, dataset, config['target_language'])

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

def get_model(config, source_vocab_size, target_vocab_size):
    """
    Creates a Seq2SeqTransformer model with specified configuration and vocabulary sizes.

    Args:
        config (dict): Configuration dictionary containing model parameters.
        source_vocab_size (int): Size of the source language vocabulary.
        target_vocab_size (int): Size of the target language vocabulary.

    Returns:
        Seq2SeqTransformer: A Seq2SeqTransformer model instance.
    """

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
    """
    Trains a Seq2SeqTransformer model.

    Args:
        config (dict): Configuration dictionary containing training parameters.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'using device {device}')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    dataset = get_dataset('iwslt2017', config)
    train_loader = dataset['train_dataloader']
    valid_loader = dataset['valid_dataloader']
    source_tokenizer = dataset['source_tokenizer']
    target_tokenizer = dataset['target_tokenizer']

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

            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        validate_model(model, valid_loader, target_tokenizer, config['max_len'], device, lambda msg: batch_iterator.write(msg))

        model_filename = get_weights_file_path(config, epoch)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step,
        }, model_filename)

def validate_model(model, validation_loader, target_tokenizer, max_len, device, print_massage, num_examples=2):
    """
    Validates the Seq2SeqTransformer model.

    Args:
        model (Seq2SeqTransformer): The trained Seq2SeqTransformer model.
        validation_loader (DataLoader): DataLoader for the validation dataset.
        target_tokenizer: Tokenizer for the target language.
        max_len (int): Maximum sequence length.
        device: Device to use for computation (cpu or cuda).
        print_message (function): Function to print messages during validation.
        num_examples (int, optional): Number of examples to validate. Defaults to 2.
    """

    model.eval()
    count = 0

    with torch.no_grad():
        for batch in validation_loader:
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)

            model_out = greedy_decode(model, encoder_input, encoder_mask, target_tokenizer, max_len, device)

            source_text = batch['source_text'][0]
            target_text = batch['target_text'][0]
            model_out_text = target_tokenizer.decode(model_out.detach().cpu().numpy())

            print_massage('-' * 80)
            print_massage(f'source: {source_text}')
            print_massage(f'target: {target_text}')
            print_massage(f'predicted: {model_out_text}')

            if count == num_examples:
                break


def greedy_decode(model: Seq2SeqTransformer, source, source_mask, target_tokenizer, max_len, device):
    """
    Greedy decoding for Seq2SeqTransformer model.

    Args:
        model (Seq2SeqTransformer): The trained Seq2SeqTransformer model.
        source: Input sequence for decoding.
        source_mask: Input sequence mask.
        target_tokenizer: Tokenizer for the target language.
        max_len (int): Maximum sequence length.
        device: Device to use for computation (cpu or cuda).

    Returns:
        torch.Tensor: Decoded output sequence.
    """

    sos_idx = target_tokenizer.token_to_id('<SOS>')
    eos_idx = target_tokenizer.token_to_id('<EOS>')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = TranslationDataset.casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        decoder_output = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        probability = model.project(decoder_output[:, -1])
        _, next_word = torch.max(probability, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)

        if next_word == eos_idx:
            break
    return decoder_input.squeeze(0)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)