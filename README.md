# Visualize_attention_mechanism
Code used for creating results published in my [Visualizing Attention Layers: Exploring Attention Heads as Dynamic Soft Weights](https://medium.com/@shantanu.darveshi/attention-layers-as-soft-weights-6bfaf125cd11) Medium article.
This project implements and analyzes an attention-based neural network that learns to perform minimum and maximum operations on pairs of numbers.

Below text is generated using Claude Sonnet 3.5 (new).
## Overview

The project consists of:
- An attention-based neural network implementation
- Training scripts for min/max operations
- Visualization tools for attention patterns
- Analysis of how the network learns to distinguish between minimum and maximum operations

## Requirements

- PyTorch
- CUDA-capable GPU
- NumPy
- Matplotlib
- Seaborn
- tqdm
- einops

## Project Structure

The main components include:
- `AttentionNetwork`: A custom neural network implementation (in utils.py)
- Training scripts for model optimization
- Visualization utilities for attention patterns
- Model weights saving and loading functionality

## Dataset

The dataset is synthetically generated with the following characteristics:
- Input range: Numbers from 0 to 127
- Two types of operations: max() and min()
- Training samples: 1/3 of all possible combinations
- Test samples: 2/3 of all possible combinations
- Format: (number1, operation_token, number2, separator_token)
  - max operation token: 128
  - min operation token: 129
  - separator token: 130

## Training

The model is trained using:
- Optimizer: AdamW
- Learning rate: 2e-3
- Weight decay: 1.0
- Loss function: CrossEntropyLoss
- Epochs: 750

Training progress is monitored through:
- Training accuracy
- Validation accuracy
- Loss metrics

## Visualization

The project includes visualization tools for:
- Training and validation accuracy curves
- Attention weights heatmaps
- Attention scores heatmaps

The visualizations are generated for both maximum and minimum operations, showing how the network attends to different input tokens.

## Usage

1. Initialize and train the model:
```python
model = AttentionNetwork().to('cuda')
# Training loop is provided in the notebook
```

2. Load pre-trained weights:
```python
model = AttentionNetwork()
model.to('cuda')
model.load_state_dict(torch.load('model_weights.pt', weights_only=True))
```

3. Generate visualizations:
```python
# Visualization code is provided in the notebook for both
# attention weights and scores
```

## Results

The attention visualizations reveal how the network:
- Develops specialized attention patterns for each operation
- Uses different attention heads to capture various aspects of the computation

## Notes

- The model uses CUDA acceleration and requires a GPU for training, replace 'cuda' with 'cpu' if you do not have access to GPU
- Visualizations use the 'YlGnBu' colormap for attention patterns
- Attention patterns are analyzed at the token level to understand the network's decision-making process
