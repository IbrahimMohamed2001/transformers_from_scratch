from pathlib import Path


def get_config():
    return {
        'batch_size': 16,
        'learning_rate': 1e-4,
        'num_epochs': 20,
        'max_len': 500,
        'd_model': 512,
        'source_language': 'ar',
        'target_language': 'en',
        'model_folder': 'weights',
        'preload': None,
        'model_basename': 'transformer_model',
        'tokenizer_file': 'tokenizer_{0}.json',
        'experiment_name': 'runs/tmodel',
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f'{model_basename}_{epoch}.pt'
    return str(Path('.') / model_folder / model_filename)