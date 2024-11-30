# Classify5: Advanced Image Classification Framework

A modular and extensible framework for image classification experiments, with support for traditional classifiers, poisoning attacks, and model analysis.

## Features

- **Modular Configuration System**: Easily configure experiments through a hierarchical configuration system
- **Multiple Dataset Support**:
  - CIFAR-100
  - GTSRB (German Traffic Sign Recognition Benchmark)
  - ImageNette
- **Model Architectures**:
  - WideResNet (28-10)
  - Custom CNN for GTSRB
  - ResNet50 for ImageNette
- **Traditional Classifiers**:
  - k-Nearest Neighbors (kNN)
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
- **Feature Extraction**: Extract deep features from pretrained models
- **Experiment Management**: Comprehensive logging and result tracking

## Installation

1. Clone the repository:

```bash
git clone https://github.com/briancphillips/classify5.git
cd classify5
```

2. Create and activate conda environment:

```bash
conda create -n classify5 python=3.10
conda activate classify5
```

3. Install dependencies:

```bash
pip install -e .
```

4. Download pretrained model checkpoint:

```bash
# Create checkpoint directory
mkdir -p checkpoints/wideresnet

# Download checkpoint
wget https://github.com/briancphillips/classify5/releases/download/v1.0/wideresnet_best.pt -O checkpoints/wideresnet/wideresnet_best.pt
```

## Project Structure

```
classify5/
├── config/                 # Configuration modules
│   ├── base_config.py     # Base configuration classes
│   ├── dataset_config.py  # Dataset-specific configurations
│   ├── model_config.py    # Model architecture configurations
│   └── experiment_config.py# Experiment configurations
├── models/                 # Model implementations
│   ├── architectures.py   # Neural network architectures
│   ├── data.py           # Dataset loading and preprocessing
│   └── factory.py        # Model creation utilities
├── experiments/           # Experiment implementations
│   └── traditional.py    # Traditional classifier experiments
├── utils/                 # Utility functions
│   └── logging.py        # Logging configuration
├── examples/              # Example scripts
│   └── traditional_example.py  # Traditional classifier example
└── checkpoints/          # Model checkpoints
    └── wideresnet/       # WideResNet checkpoints
        └── wideresnet_best.pt  # Pretrained WideResNet-28-10
```

## Quick Start

Run a traditional classifier experiment on CIFAR-100:

```python
from config.experiment_config import create_experiment_config
from experiments.traditional import TraditionalExperiment

# Create experiment
exp = TraditionalExperiment(
    config=create_experiment_config(
        dataset_name="cifar100",
        model_name="wrn-28-10",
        checkpoint_path="checkpoints/wideresnet/wideresnet_best.pt",
    )
)

# Run experiment
exp.run()
```

Or use the example script:

```bash
python examples/traditional_example.py
```

## Configuration

The framework uses a hierarchical configuration system:

1. **Dataset Configuration**: Configure dataset parameters

   - Dataset name
   - Batch size
   - Number of workers
   - Data augmentation

2. **Model Configuration**: Configure model architecture

   - Model name
   - Checkpoint path
   - Architecture-specific parameters

3. **Hardware Configuration**: Configure hardware settings
   - Number of workers
   - Pin memory
   - Device (CPU/GPU)

Example configuration:

```python
config = create_experiment_config(
    dataset_name="cifar100",
    model_name="wrn-28-10",
    checkpoint_path="path/to/checkpoint.pt",
    batch_size=128,
    num_workers=4,
    pin_memory=True,
)
```

## Results

Results are saved in the following format:

- Experiment logs: `logs/experiment.log`
- Model checkpoints: `checkpoints/<model_name>/`
- Results: `results/<experiment_name>/`

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- WideResNet implementation based on [paper](https://arxiv.org/abs/1605.07146)
- CIFAR-100 dataset from [paper](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
