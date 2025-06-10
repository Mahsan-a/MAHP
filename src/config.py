
"""
Configuration settings for the glucose prediction project.
"""
import os

# Data paths
ADULT_DATA_PATH = PATH
PEDIATRIC_DATA_PATH = PATH

# Model architecture parameters
MATRIX_SHAPE0 = 288
DAILY_SHAPE = 327
INDIVIDUAL_SHAPE = 324

# Training parameters
DEFAULT_CONFIG = {
    'num_splits': 8,
    'num_epochs': 80,
    'batch_size': 32,
    'patience': 10,
    'learning_rate': 2e-4
}

# Label definitions
LABEL_NAMES = [
    "hypo_early_night", "hypo_night", "hypo_long_night",
    "hypo_night_morning", "hyper_day", "hypo_late_night",
    "hyper_night", "hyper_early_night", "hypo_morning", "hypo_day"
]

