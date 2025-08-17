#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main CLI interface for TGN-SVDD experiments.

This script provides a command-line interface for running TGN-SVDD intrusion
detection experiments with configurable parameters.
"""

import sys
import warnings
import logging
import os
from pathlib import Path

# Suppress matplotlib warnings early
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='Glyph.*missing from current font')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

# Add src directory to path for imports
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Now import everything normally
from config.experiment_config import (
    TGNSVDDConfig, 
    create_quick_test_config, 
    parse_args, 
    merge_cli_args, 
    validate_config,
    print_config_summary
)
from experiments.tgn_svdd_experiment import run_multi_day_experiments, run_single_day_experiment


def main():
    """Main entry point for TGN-SVDD experiments."""
    print("🚀 TGN-SVDD Intrusion Detection Experiments")
    print("=" * 60)
    
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Create configuration
        if args.quick_test:
            config = create_quick_test_config()
        else:
            config = TGNSVDDConfig()
        
        # Merge CLI arguments
        config = merge_cli_args(config, args)
        
        # Validate configuration
        validation_errors = validate_config(config)
        if validation_errors:
            print("❌ Configuration validation failed:")
            for error in validation_errors:
                print(f"   - {error}")
            sys.exit(1)
        
        # Print configuration summary
        print_config_summary(config)
        
        # Confirm before running
        if not args.quick_test and len(config.days_to_run) > 1:
            response = input("\nProceed with multi-day experiment? [y/N]: ")
            if response.lower() not in ['y', 'yes']:
                print("Experiment cancelled.")
                sys.exit(0)
        
        # Run experiments
        print("\n🔬 Starting experiments...")
        
        if len(config.days_to_run) == 1:
            # Single day experiment
            day = config.days_to_run[0]
            results = run_single_day_experiment(config, day)
            
            if 'error' in results:
                print(f"❌ Experiment failed: {results['error']}")
                sys.exit(1)
            else:
                print(f"✅ Experiment completed successfully for {day}")
                
                # Print summary metrics
                final_metrics = results.get('metrics', {})
                if final_metrics:
                    final_epoch = max(final_metrics.keys())
                    metrics = final_metrics[final_epoch]
                    print(f"Final F1 Score: {metrics['f1_score']:.4f}")
                    print(f"Final Threshold: {metrics['threshold']:.4f}")
        
        else:
            # Multi-day experiments
            all_results = run_multi_day_experiments(config)
            
            # Print summary
            print("\n📊 Experiment Summary:")
            print("-" * 40)
            
            successful_days = []
            failed_days = []
            
            for day, results in all_results.items():
                if 'error' in results:
                    failed_days.append(day)
                    print(f"❌ {day}: FAILED - {results['error']}")
                else:
                    successful_days.append(day)
                    # Get final metrics if available
                    final_metrics = results.get('metrics', {})
                    if final_metrics:
                        final_epoch = max(final_metrics.keys())
                        f1_score = final_metrics[final_epoch]['f1_score']
                        print(f"✅ {day}: F1 = {f1_score:.4f}")
                    else:
                        print(f"✅ {day}: COMPLETED")
            
            print(f"\nResults: {len(successful_days)} successful, {len(failed_days)} failed")
        
        print("\n🎉 All experiments completed!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
