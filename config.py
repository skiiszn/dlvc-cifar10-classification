"""
Central configuration for machine-dependent paths used in assignment scripts.

Students should adapt these paths to their local setup.
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent

# Path to extracted CIFAR-10 python files (directory containing data_batch_1 ... test_batch)
DATA_DIR = BASE_DIR / "cifar-10-batches-py"

# Optional logging directory (for wandb/tensorboard/custom logs)
LOG_DIR = BASE_DIR / "logs"

# Directory where trained model checkpoints are stored
MODEL_SAVE_DIR = BASE_DIR / "checkpoints"
