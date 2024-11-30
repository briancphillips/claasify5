# Technical Documentation

## Architecture Overview

### Core Components

1. **Configuration System** (`config/`)

   - `base_config.py`: Base configuration classes and utilities
   - `dataset_config.py`: Dataset-specific configurations
   - `model_config.py`: Model architecture configurations
   - `experiment_config.py`: Experiment configurations

2. **Models** (`models/`)

   - `architectures.py`: Neural network implementations
     - WideResNet-28-10
     - Custom CNN for GTSRB
     - ResNet50 for ImageNette
   - `data.py`: Dataset loading and preprocessing
   - `factory.py`: Model creation utilities

3. **Experiments** (`experiments/`)

   - `traditional.py`: Traditional classifier experiments
     - Feature extraction
     - Classifier training and evaluation
     - Results logging

4. **Utils** (`utils/`)
   - `logging.py`: Logging configuration and utilities

### Data Flow

1. **Configuration Creation**

   ```python
   config = create_experiment_config(
       dataset_name="cifar100",
       model_name="wrn-28-10",
       checkpoint_path="checkpoints/wideresnet/wideresnet_best.pt"
   )
   ```

2. **Experiment Initialization**

   ```python
   experiment = TraditionalExperiment(config=config)
   ```

3. **Data Loading**

   - Dataset loading through PyTorch DataLoader
   - Automatic batch processing
   - Optional data augmentation

4. **Feature Extraction**

   - Load pretrained model
   - Extract features from intermediate layers
   - Cache features for faster subsequent access

5. **Classifier Training**

   - Train traditional classifiers on extracted features
   - Support for multiple classifier types
   - Cross-validation and hyperparameter tuning

6. **Results Collection**
   - Metrics calculation
   - Results logging
   - Optional visualization

## Implementation Details

### Model Architecture

#### WideResNet-28-10

- 28 layers deep
- Widening factor of 10
- Batch normalization
- ReLU activation
- Dropout rate: 0.3

```python
class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, dropout_rate=0.3, num_classes=100):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(WideBasicBlock, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasicBlock, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasicBlock, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)
```

### Feature Extraction

Features are extracted from the penultimate layer:

```python
def extract_features(self, x):
    out = self.conv1(x)
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = F.relu(self.bn1(out))
    out = F.avg_pool2d(out, 8)
    return out.view(out.size(0), -1)
```

### Traditional Classifiers

1. **k-Nearest Neighbors (kNN)**

   - Default k: 5
   - Distance metric: Euclidean
   - Weighted voting: distance-weighted

2. **Logistic Regression**

   - L2 regularization
   - Multi-class: One-vs-Rest
   - Max iterations: 1000

3. **Random Forest**

   - Number of trees: 100
   - Max depth: None
   - Min samples split: 2
   - Min samples leaf: 1

4. **Support Vector Machine (SVM)**
   - Kernel: RBF
   - C: 1.0
   - Multi-class: One-vs-One

## Configuration Options

### Dataset Configuration

```python
@dataclass
class DatasetConfig:
    name: str
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    data_dir: str = "data"
    train_transforms: Optional[List[str]] = None
    test_transforms: Optional[List[str]] = None
```

### Model Configuration

```python
@dataclass
class ModelConfig:
    name: str
    checkpoint_path: str
    num_classes: int = 100
    in_channels: int = 3
    pretrained: bool = True
    device: str = "cuda"
```

### Experiment Configuration

```python
@dataclass
class ExperimentConfig:
    dataset: DatasetConfig
    model: ModelConfig
    classifiers: List[str] = field(default_factory=lambda: ["knn", "lr", "rf", "svm"])
    feature_dim: int = 640
    pca_components: Optional[int] = 512
    results_dir: str = "results"
    log_file: str = "logs/experiment.log"
```

## Logging System

The framework uses Python's built-in logging module with custom formatting:

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.log_file),
        logging.StreamHandler()
    ]
)
```

## Results Format

Results are saved in JSON format with the following structure:

```json
{
    "experiment_name": "traditional_cifar100",
    "dataset": "cifar100",
    "model": "wrn-28-10",
    "timestamp": "2023-11-30-12-00-00",
    "metrics": {
        "knn": {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0
        },
        "lr": {...},
        "rf": {...},
        "svm": {...}
    },
    "config": {...},
    "runtime": {
        "feature_extraction": 0.0,
        "training": 0.0,
        "inference": 0.0
    }
}
```
