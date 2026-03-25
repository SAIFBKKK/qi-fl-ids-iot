# Contributing to Quantum-Inspired Federated IDS for IoT

First off, thank you for considering contributing to this project! It's people like you that make this such a great tool.

---

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

---

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps which reproduce the problem** in as many details as possible
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed after following the steps**
- **Explain which behavior you expected to see instead and why**
- **Include screenshots and animated GIFs if possible**
- **Include your environment details** (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please include:

- **Use a clear and descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Provide specific examples to demonstrate the steps**
- **Describe the current behavior and expected behavior**
- **Explain why this enhancement would be useful**

### Pull Requests

- Fill in the required template
- Follow the Python/code styleguides
- Include appropriate test cases
- Update documentation as needed
- Ensure all tests pass locally before submitting

---

## Development Setup

### Prerequisites

```bash
# Check Python version
python --version  # 3.9 or higher

# Clone repository
git clone https://github.com/SAIFBKKK/Quantum-Inspired-Federated-IDS-FOR-IOT.git
cd Quantum-Inspired-Federated-IDS-FOR-IOT
```

### Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development tools

# Install pre-commit hooks
pre-commit install
```

### Development Tools

The following tools are configured for code quality:

- **black** — Code formatting
- **flake8** — Linting
- **isort** — Import sorting
- **mypy** — Type checking
- **pytest** — Testing
- **pytest-cov** — Coverage reporting

```bash
# Format code
black src/ experiments/

# Check style
flake8 src/ experiments/ --max-line-length=100

# Type checking
mypy src/

# Run tests
pytest tests/ -v

# Coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## Code Style Guide

### Python Code Style

We follow **PEP 8** with these specific guidelines:

#### 1. File Header

```python
"""
Module description: Brief description of what this module does.

This module handles [specific functionality].

Author: Your Name
Date: YYYY-MM-DD
Version: 1.0
"""
```

#### 2. Imports

```python
# Standard library imports first
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Third-party imports
import numpy as np
import torch
import flwr as fl

# Local imports
from src.common.logger import get_logger
from src.data.loader import DataLoader
```

#### 3. Type Hints

All functions must include type hints:

```python
def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epochs: int = 5,
    learning_rate: float = 0.001,
) -> Tuple[float, float]:
    """
    Train a PyTorch model.
    
    Args:
        model: The neural network model to train
        train_loader: DataLoader for training data
        epochs: Number of training epochs (default: 5)
        learning_rate: Learning rate for optimizer (default: 0.001)
    
    Returns:
        Tuple of (final_loss, final_accuracy)
    
    Raises:
        ValueError: If epochs is negative
        TypeError: If model is not a PyTorch module
    
    Example:
        >>> model = IDS_Model(input_size=33, hidden_size=128)
        >>> loss, acc = train_model(model, train_loader, epochs=10)
    """
    if epochs < 0:
        raise ValueError(f"epochs must be non-negative, got {epochs}")
    
    # Implementation
    return final_loss, final_accuracy
```

#### 4. Docstrings

Use Google-style docstrings:

```python
def federated_average(weights_list: List[np.ndarray]) -> np.ndarray:
    """
    Compute federated averaging of model weights.
    
    Implements the FedAvg algorithm from McMahan et al. (2017).
    Each client's weights are averaged equally.
    
    Args:
        weights_list: List of weight arrays from clients
    
    Returns:
        Averaged weight array
    
    Raises:
        ValueError: If weights_list is empty
        TypeError: If weights have incompatible shapes
    
    References:
        McMahan, H. B., Moore, E., Ramage, D., & Aguerri, Y. A. (2017).
        Communication-Efficient Learning of Deep Networks from Decentralized Data.
        Proceedings of the AISTATS Conference.
    """
    if not weights_list:
        raise ValueError("weights_list cannot be empty")
    
    return np.mean(weights_list, axis=0)
```

#### 5. Constants

Use UPPER_CASE for constants:

```python
# Configuration constants
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 0.001
MAX_TRAINING_ROUNDS = 100
NUM_ATTACK_CLASSES = 34

# File paths
DATA_DIR = Path(__file__).parent / "data"
CONFIG_DIR = Path(__file__).parent / "configs"
```

#### 6. Class Structure

```python
class FederatedServer:
    """Federated Learning server using Flower framework.
    
    Attributes:
        num_rounds: Total number of training rounds
        num_clients: Number of participating clients
        strategy: FL aggregation strategy (FedAvg, etc.)
    """
    
    def __init__(self, num_rounds: int = 25, num_clients: int = 3):
        """Initialize the FL server.
        
        Args:
            num_rounds: Number of training rounds
            num_clients: Number of clients to expect
        
        Raises:
            ValueError: If num_rounds or num_clients is invalid
        """
        self.num_rounds = num_rounds
        self.num_clients = num_clients
        self._validate_params()
    
    def _validate_params(self) -> None:
        """Validate server parameters."""
        if self.num_rounds <= 0:
            raise ValueError("num_rounds must be positive")
        if self.num_clients <= 0:
            raise ValueError("num_clients must be positive")
    
    def start(self) -> None:
        """Start the federated learning server."""
        pass
```

### Line Length

- Maximum **100 characters** per line
- Use black for automatic formatting: `black src/`

### Naming Conventions

```python
# Functions and variables: snake_case
def load_data_partition(node_id: str) -> Dict:
    pass

# Classes: PascalCase
class FederatedLearningClient:
    pass

# Constants: UPPER_CASE
NUM_ROUNDS = 25
DEFAULT_PORT = 8080

# Private methods/attributes: _leading_underscore
def _prepare_model_weights(self) -> None:
    pass
```

---

## Testing

### Writing Tests

Test files should be placed in the `tests/` directory and follow naming convention `test_*.py`.

```python
import pytest
import torch
from src.models.ids_model import IDS_Model
from src.data.loader import DataLoader


class TestIDSModel:
    """Test suite for IDS model."""
    
    @pytest.fixture
    def model(self):
        """Create a test model."""
        return IDS_Model(input_size=33, hidden_size=128)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        return torch.randn(32, 33)  # batch_size=32, features=33
    
    def test_model_initialization(self, model):
        """Test that model initializes correctly."""
        assert model is not None
        assert isinstance(model, torch.nn.Module)
    
    def test_forward_pass(self, model, sample_data):
        """Test forward pass through model."""
        output = model(sample_data)
        assert output.shape == (32, 5)  # 5 attack classes
    
    def test_model_output_range(self, model, sample_data):
        """Test that model outputs are valid probabilities."""
        output = model(sample_data)
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)


def test_data_loader_shapes():
    """Test that DataLoader returns correct tensor shapes."""
    loader = DataLoader(batch_size=256)
    X, y = loader.get_batch()
    
    assert X.shape[1] == 33  # Feature dimension
    assert y.shape[0] == X.shape[0]  # Same batch size


@pytest.mark.parametrize("batch_size", [32, 64, 128, 256])
def test_different_batch_sizes(batch_size):
    """Test DataLoader with different batch sizes."""
    loader = DataLoader(batch_size=batch_size)
    X, _ = loader.get_batch()
    assert X.shape[0] == batch_size
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run specific test
pytest tests/test_models.py::TestIDSModel::test_forward_pass -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run tests matching pattern
pytest tests/ -k "test_forward" -v
```

### Test Coverage Requirements

- Minimum 80% code coverage for new functionality
- All critical paths must be tested
- Use `pytest --cov` to check coverage

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Git Workflow

### Branching Strategy

We use GitHub Flow:

1. **main** — Stable, production-ready code
2. **develop** — Integration branch
3. **feature/*** — New features
4. **bugfix/*** — Bug fixes
5. **docs/*** — Documentation updates

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body>

<footer>
```

#### Type

- `feat` — New feature
- `fix` — Bug fix
- `docs` — Documentation
- `style` — Code style (formatting, missing semicolons, etc.)
- `refactor` — Code refactoring
- `perf` — Performance improvement
- `test` — Adding tests
- `chore` — Build process, dependencies

#### Examples

```
feat(federated-learning): implement FedAvg aggregation strategy

- Add FedAvgStrategy class with proper weight averaging
- Support client dropout during aggregation
- Add comprehensive tests for edge cases

Fixes #42
```

```
fix(data-loader): resolve memory leak in batch processing

Release large arrays after processing to prevent memory buildup.
```

```
docs(readme): update installation instructions for Docker
```

### Pull Request Process

1. Create feature branch from `develop`:
   ```bash
   git checkout -b feature/your-feature-name develop
   ```

2. Make changes and commit with proper messages:
   ```bash
   git commit -m "feat: your feature description"
   ```

3. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Create Pull Request with:
   - Clear title and description
   - Reference to related issues
   - Screenshots/examples if applicable
   - Summary of changes

5. Ensure CI/CD checks pass:
   - All tests passing
   - Code style checks passing
   - No coverage regression

6. Request review from maintainers

7. After approval, merge and delete branch

---

## Documentation

### Docstring Standards

All modules, classes, and functions must have docstrings following Google style:

```python
def calculate_f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> float:
    """
    Calculate F1 score for classification metrics.
    
    Supports different averaging strategies for multi-class problems.
    
    Args:
        y_true: True class labels (shape: [n_samples,])
        y_pred: Predicted class labels (shape: [n_samples,])
        average: Averaging method ('macro', 'micro', 'weighted')
    
    Returns:
        F1 score as float between 0 and 1
    
    Raises:
        ValueError: If y_true and y_pred have different lengths
        ValueError: If average method is not supported
    
    Note:
        Uses sklearn.metrics.f1_score internally.
        For imbalanced datasets, macro averaging is recommended.
    
    Example:
        >>> y_true = np.array([0, 1, 2, 0, 1, 2])
        >>> y_pred = np.array([0, 1, 2, 0, 1, 1])
        >>> f1 = calculate_f1_score(y_true, y_pred, average='macro')
        >>> print(f"{f1:.3f}")
        0.889
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    
    # Implementation
    return f1_score(y_true, y_pred, average=average)
```

### README Standards

- Keep README focused and well-organized
- Include clear examples
- Link to detailed documentation
- Update when adding major features

---

## Code Review Checklist

When submitting a PR, ensure:

- [ ] Code follows style guide (PEP 8, type hints, docstrings)
- [ ] All tests pass locally
- [ ] New tests added for new functionality
- [ ] Coverage >= 80%
- [ ] Commit messages follow conventional commits
- [ ] Documentation updated (if needed)
- [ ] No breaking changes (or documented in migration guide)
- [ ] Related issues referenced

---

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes for their contributions
- GitHub contributors page

---

## Questions?

- Open an issue on GitHub for questions
- Check existing issues before asking
- Use GitHub Discussions for broader topics
- Contact maintainers for security issues (see SECURITY.md)

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing! Your work helps make this project better for everyone.
