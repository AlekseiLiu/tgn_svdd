#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration management for TGN-SVDD experiments.

This module provides configuration parameters and CLI argument parsing
for the TGN-SVDD experiment pipeline.
"""

import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class TGNSVDDConfig:
    """Configuration parameters for TGN-SVDD experiments."""
    
    # Data settings
    #data_split: str = 'cic_2017_monday_0_0_1'
    days_to_run: List[str] = field(default_factory=lambda: [
        'monday_friday_workinghours',
        'monday_thursday_workinghours', 
        'monday_tuesday_workinghours',
        'monday_wednesday_workinghours'
    ])
    
    # Model hyperparameters
    memory_dim: int = 200
    embedding_dim: int = 200
    time_dim: int = 200
    neighbor_size: int = 10
    
    # Training settings
    n_epochs: int = 30
    batch_size: int = 200
    learning_rate: float = 0.0001
    regularization_weight: float = 1e-6
    
    # Data splits
    val_ratio: float = 0.15
    test_ratio: float = 0.54
    
    # Evaluation
    evaluation_epochs: List[int] = field(default_factory=lambda: [10, 20, 25, 30])
    threshold_percentile: int = 99
    
    # Output settings
    save_checkpoints: bool = False
    results_dir: str = "results"
    seed: int = 12345
    
    # Experimental settings (for quick testing)
    quick_test: bool = False  # If True, runs with reduced epochs and single day


def create_quick_test_config() -> TGNSVDDConfig:
    """Create a configuration optimized for quick testing."""
    config = TGNSVDDConfig()
    config.quick_test = True
    config.n_epochs = 5
    config.days_to_run = ['monday_friday_workinghours']
    config.evaluation_epochs = [3, 5]
    return config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for TGN-SVDD experiments."""
    parser = argparse.ArgumentParser(
        description="TGN-SVDD Intrusion Detection Experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to config YAML file (not implemented yet)"
    )
    parser.add_argument(
        "--day", 
        type=str, 
        choices=['monday_friday_workinghours', 'monday_thursday_workinghours', 
                'monday_tuesday_workinghours', 'monday_wednesday_workinghours'],
        help="Run specific day only"
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with reduced epochs and single day"
    )
    
    # Training overrides
    parser.add_argument(
        "--epochs", 
        type=int, 
        help="Override number of epochs"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        help="Override batch size"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        help="Override learning rate"
    )
    
    # Output settings
    parser.add_argument(
        "--results-dir", 
        type=str, 
        help="Override results directory"
    )
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save model checkpoints during training"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def merge_cli_args(config: TGNSVDDConfig, args: argparse.Namespace) -> TGNSVDDConfig:
    """Merge CLI arguments into configuration object."""
    
    # Handle quick test mode
    if args.quick_test:
        config = create_quick_test_config()
    
    # Override with CLI arguments if provided
    if args.day:
        config.days_to_run = [args.day]
    
    if args.epochs:
        config.n_epochs = args.epochs
        
    if args.batch_size:
        config.batch_size = args.batch_size
        
    if args.lr:
        config.learning_rate = args.lr
        
    if args.results_dir:
        config.results_dir = args.results_dir
        
    if args.save_checkpoints:
        config.save_checkpoints = True
    
    return config


def validate_config(config: TGNSVDDConfig) -> List[str]:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration object to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Validate epochs
    if config.n_epochs <= 0:
        errors.append(f"n_epochs must be positive, got: {config.n_epochs}")
    
    # Validate batch size
    if config.batch_size <= 0:
        errors.append(f"batch_size must be positive, got: {config.batch_size}")
    
    # Validate learning rate
    if config.learning_rate <= 0:
        errors.append(f"learning_rate must be positive, got: {config.learning_rate}")
    
    # Validate ratios
    if not (0.0 <= config.val_ratio <= 1.0):
        errors.append(f"val_ratio must be between 0.0 and 1.0, got: {config.val_ratio}")
    
    if not (0.0 <= config.test_ratio <= 1.0):
        errors.append(f"test_ratio must be between 0.0 and 1.0, got: {config.test_ratio}")
    
    if config.val_ratio + config.test_ratio > 1.0:
        errors.append(f"val_ratio + test_ratio cannot exceed 1.0, got: {config.val_ratio + config.test_ratio}")
    
    # Validate threshold percentile
    if not (0 <= config.threshold_percentile <= 100):
        errors.append(f"threshold_percentile must be between 0 and 100, got: {config.threshold_percentile}")
    
    # Validate days to run
    valid_days = ['monday_friday_workinghours', 'monday_thursday_workinghours', 
                  'monday_tuesday_workinghours', 'monday_wednesday_workinghours']
    for day in config.days_to_run:
        if day not in valid_days:
            errors.append(f"Invalid day: {day}. Must be one of: {valid_days}")
    
    return errors


def print_config_summary(config: TGNSVDDConfig) -> None:
    """Print configuration summary."""
    print("=" * 60)
    print("TGN-SVDD Experiment Configuration")
    print("=" * 60)
    print(f"Data split: {config.data_split}")
    print(f"Days to run: {config.days_to_run}")
    print(f"Epochs: {config.n_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Memory dim: {config.memory_dim}")
    print(f"Embedding dim: {config.embedding_dim}")
    print(f"Results dir: {config.results_dir}")
    print(f"Save checkpoints: {config.save_checkpoints}")
    if config.quick_test:
        print("🚀 QUICK TEST MODE ENABLED")
    print("=" * 60)


if __name__ == "__main__":
    # Test the configuration system
    print("Testing TGN-SVDD Configuration System")
    
    # Test default config
    config = TGNSVDDConfig()
    print("\nDefault Configuration:")
    print_config_summary(config)
    
    # Test validation
    errors = validate_config(config)
    if errors:
        print(f"Validation errors: {errors}")
    else:
        print("✅ Configuration is valid")
    
    # Test quick test config
    quick_config = create_quick_test_config()
    print("\nQuick Test Configuration:")
    print_config_summary(quick_config)
