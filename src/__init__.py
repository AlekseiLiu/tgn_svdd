#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Package initialization and import helper for TGN-SVDD.

This script helps resolve import issues when running scripts directly
by setting up the proper Python path.
"""

import sys
import os

# Add the src directory to Python path
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Export key components for easy import
__all__ = [
    'TGNSVDDConfig',
    'create_quick_test_config',
    'TGNDataLoader',
    'TGNSVDDModelFactory',
    'TGNSVDDTrainer',
    'TGNSVDDEvaluator',
    'TGNSVDDExperimentRefactored'
]

# Try to import key components
try:
    from config.experiment_config import TGNSVDDConfig, create_quick_test_config
    from data.data_loader import TGNDataLoader
    from models.model_factory import TGNSVDDModelFactory
    from training.trainer import TGNSVDDTrainer
    from training.evaluator import TGNSVDDEvaluator
    from experiments.tgn_svdd_experiment_refactored import TGNSVDDExperimentRefactored
    
    print("✅ TGN-SVDD package initialized successfully")
    
except ImportError as e:
    print(f"⚠️  Some components could not be imported: {e}")
    print("This is expected if PyTorch is not available")

def get_package_info():
    """Get information about the TGN-SVDD package."""
    info = {
        'src_directory': src_dir,
        'python_path': sys.path[:3],  # First 3 entries
        'available_modules': []
    }
    
    # Check which modules are available
    modules_to_check = [
        'config.experiment_config',
        'data.data_loader', 
        'models.model_factory',
        'training.trainer',
        'training.evaluator',
        'experiments.tgn_svdd_experiment_refactored'
    ]
    
    for module in modules_to_check:
        try:
            __import__(module)
            info['available_modules'].append(module)
        except ImportError:
            pass
    
    return info

if __name__ == "__main__":
    print("🏗️  TGN-SVDD Package Information")
    print("=" * 40)
    
    info = get_package_info()
    print(f"Source directory: {info['src_directory']}")
    print(f"Python path entries: {info['python_path']}")
    print(f"Available modules: {len(info['available_modules'])}")
    
    for module in info['available_modules']:
        print(f"  ✅ {module}")
    
    if len(info['available_modules']) >= 6:
        print("\n🎉 All core modules are available!")
    else:
        print("\n⚠️  Some modules are missing (likely due to PyTorch dependencies)")
    
    print("\n📋 Usage:")
    print("  import __init__  # Sets up paths automatically")
    print("  from config.experiment_config import TGNSVDDConfig")
