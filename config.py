from pathlib import Path


def get_config():
    """
    Get the configuration parameters for the translation model.

    Returns:
        dict: Configuration dictionary containing the parameters.
    """

    return {
        'batch_size': 16,
        'learning_rate': 1e-4,
        'num_layers': 6,
        'heads': 8,
        'dropout': 0.1,
        'hidden_size_ff': 2048,
        'num_epochs': 1,
        'max_len': 500,
        'd_model': 512,
        'source_language': 'en',
        'target_language': 'ar',
        'model_folder': 'weights',
        'preload': None,
        'model_basename': 'transformer_model',
        'tokenizer_file': 'tokenizer_{0}.json',
        'experiment_name': 'runs/tmodel',
    }

def get_weights_file_path(config, epoch: str):
    """
    Get the file path for saving or loading model weights for a specific epoch.

    Args:
        config (dict): Configuration dictionary containing parameters.
        epoch (str): Epoch number as a string.

    Returns:
        str: File path for the model weights file.
    """

    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f'{model_basename}_{epoch}.pt'
    return str(Path('.') / model_folder / model_filename)