# Transformer from Scratch with PyTorch

This repository contains an implementation of the Transformer architecture from scratch using PyTorch. It includes scripts for training the model, tracking the training process, and logging values using TensorBoard.

## Table of Contents

- [Transformer from Scratch with PyTorch](#transformer-from-scratch-with-pytorch)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Configuration](#configuration)
    - [Training](#training)
    - [Validation](#validation)
    - [Inference](#inference)
  - [Configuration](#configuration-1)
  - [Acknowledgements](#acknowledgements)

## Overview

The Transformer model is a deep learning architecture introduced in the paper "Attention is All You Need" by Vaswani et al. It has become the foundation for many state-of-the-art models in natural language processing (NLP), such as BERT and GPT.

This repository provides a complete implementation of the Transformer model, including:

- Tokenization and dataset preparation
- Model architecture (embedding, encoder, decoder, multi-head attention, etc.)
- Training loop with logging to TensorBoard
- Validation and inference scripts

## Installation

To get started, clone the repository and install the required dependencies:

``` bash
git clone https://github.com/IbrahimMohamed2001/transformers-from-scratch.git
cd transformers-from-scratch
pip install -r requirements.txt
```

## Usage

### Configuration

The configuration parameters for the model and training process are defined in `config.py`. You can modify these parameters to suit your needs.

### Training

To train the model, run the `train.py` script:

```bash
python train.py
```

This script will:

1. Load and preprocess the dataset
2. Initialize the model and optimizer
3. Train the model for the specified number of epochs
4. Save the model weights and log training metrics to TensorBoard

### Validation

The `train.py` script includes a validation step that evaluates the model on a validation dataset after each epoch. The validation results are also logged to TensorBoard.

### Inference

To perform inference with the trained model, you can use the `inference.ipynb` Jupyter notebook. This notebook provides an example of how to load the trained model and generate translations for new input sentences.

## Configuration

The `config.py` file contains the configuration parameters for the model and training process. Key parameters include:

- `batch_size`: Batch size for training
- `learning_rate`: Learning rate for the optimizer
- `num_layers`: Number of encoder and decoder layers
- `heads`: Number of attention heads
- `dropout`: Dropout rate
- `hidden_size_ff`: Hidden size of the feed-forward network
- `num_epochs`: Number of training epochs
- `max_len`: Maximum sequence length
- `d_model`: Dimensionality of the model
- `source_language`: Source language identifier
- `target_language`: Target language identifier
- `model_folder`: Folder to save model weights
- `preload`: Path to pre-trained model weights (if any)
- `model_basename`: Base name for model weight files
- `tokenizer_file`: File name pattern for tokenizers
- `experiment_name`: Name for the TensorBoard experiment

## Acknowledgements

This implementation is inspired by the original Transformer paper "Attention is All You Need" by Vaswani et al. and various open-source implementations available online.

If you find this repository useful, please consider giving it a star!
