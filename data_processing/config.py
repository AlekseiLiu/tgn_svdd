#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration management for CIC-2017 data processing.

This module provides configuration parameters and CLI argument parsing
for the CIC-2017 dataset preprocessing pipeline.
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class ProcessingConfig:
    """Configuration parameters for CIC-2017 data processing."""
    
    # Data paths
    raw_data_dir: str = "../data/raw/cic_2017/"
    output_dir: str = "../data/"
    
    # NFStreamer parameters
    idle_timeout: int = 1000
    active_timeout: int = 3000
    
    # Data splitting parameters
    val_ratio: float = 0.0
    test_ratio: float = 0.0
    
    # Processing parameters
    time_col: str = "bidirectional_first_seen_ms"
    src_col: str = "src_ip"
    dst_col: str = "dst_ip"
    
    # Feature engineering parameters
    col_drop_list: List[str] = None
    col_to_scale: List[str] = None
    
    def __post_init__(self):
        """Validate and normalize paths after initialization."""
        # Convert to absolute paths relative to the project root
        self.raw_data_dir = os.path.abspath(self.raw_data_dir)
        self.output_dir = os.path.abspath(self.output_dir)
        
        # Set default column lists if not provided
        if self.col_drop_list is None:
            self.col_drop_list = [
                'id', 'expiration_id', 'src_mac', 'src_oui', 'src_port', 'dst_mac', 'dst_oui', 'dst_port', 'protocol', 'vlan_id', 'application_name',
                'application_confidence', 'requested_server_name', 'client_fingerprint', 'server_fingerprint', 'user_agent',
                'src2dst_first_seen_ms', 'application_is_guessed', 'bidirectional_last_seen_ms', 'content_type', 'src2dst_last_seen_ms', 'dst2src_last_seen_ms'
            ]
        
        if self.col_to_scale is None:
            self.col_to_scale = [
                'bidirectional_duration_ms', 'bidirectional_packets', 'bidirectional_bytes', 'src2dst_duration_ms', 'src2dst_packets', 'src2dst_bytes',
                'dst2src_first_seen_ms', 'dst2src_duration_ms', 'dst2src_packets', 'dst2src_bytes', 'bidirectional_min_ps', 'bidirectional_mean_ps', 'bidirectional_stddev_ps',
                'bidirectional_max_ps', 'src2dst_min_ps', 'src2dst_mean_ps', 'src2dst_stddev_ps', 'src2dst_max_ps', 'dst2src_min_ps', 'dst2src_mean_ps', 'dst2src_stddev_ps',
                'dst2src_max_ps', 'bidirectional_min_piat_ms', 'bidirectional_mean_piat_ms', 'bidirectional_stddev_piat_ms', 'bidirectional_max_piat_ms', 'src2dst_min_piat_ms',
                'src2dst_mean_piat_ms', 'src2dst_stddev_piat_ms', 'src2dst_max_piat_ms', 'dst2src_min_piat_ms', 'dst2src_mean_piat_ms', 'dst2src_stddev_piat_ms', 'dst2src_max_piat_ms',
                'bidirectional_syn_packets', 'bidirectional_cwr_packets', 'bidirectional_ece_packets', 'bidirectional_urg_packets', 'bidirectional_ack_packets',
                'bidirectional_psh_packets', 'bidirectional_rst_packets', 'bidirectional_fin_packets', 'src2dst_syn_packets', 'src2dst_cwr_packets', 'src2dst_ece_packets',
                'src2dst_urg_packets', 'src2dst_ack_packets', 'src2dst_psh_packets', 'src2dst_rst_packets', 'src2dst_fin_packets', 'dst2src_syn_packets', 'dst2src_cwr_packets',
                'dst2src_ece_packets', 'dst2src_urg_packets', 'dst2src_ack_packets', 'dst2src_psh_packets', 'dst2src_rst_packets', 'dst2src_fin_packets'
            ]


def parse_arguments() -> ProcessingConfig:
    """
    Parse command line arguments and return configuration.
    
    Returns:
        ProcessingConfig: Configuration object with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Process CIC-2017 PCAP files into CSV format for TGN-SVDD experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data paths
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default="../data/raw/cic_2017/",
        help="Directory containing raw CIC-2017 PCAP files"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        default="../data/cic_2017_processing/",
        help="Directory to save processed CSV files"
    )
    
    # NFStreamer parameters
    parser.add_argument(
        "--idle-timeout",
        type=int,
        default=1000,
        help="NFStreamer idle timeout in milliseconds"
    )
    
    parser.add_argument(
        "--active-timeout",
        type=int, 
        default=3000,
        help="NFStreamer active timeout in milliseconds"
    )
    
    # Data splitting parameters
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="Validation set ratio (0.0-1.0)"
    )
    
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.0,
        help="Test set ratio (0.0-1.0)"
    )
    
    # Utility flags
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and show what would be processed without actual processing"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for detailed progress"
    )
    
    args = parser.parse_args()
    
    # Create config object from parsed arguments
    config = ProcessingConfig(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        idle_timeout=args.idle_timeout,
        active_timeout=args.active_timeout,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    return config, args.dry_run, args.verbose


def validate_config(config: ProcessingConfig) -> List[str]:
    """
    Validate configuration parameters and paths.
    
    Args:
        config: Configuration object to validate
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check raw data directory exists
    if not os.path.exists(config.raw_data_dir):
        errors.append(f"Raw data directory does not exist: {config.raw_data_dir}")
    
    # Check for required PCAP files
    required_files = [
        "Monday-WorkingHours.pcap",
        "Tuesday-WorkingHours.pcap", 
        "Wednesday-WorkingHours.pcap",
        "Thursday-WorkingHours.pcap",
        "Friday-WorkingHours.pcap"
    ]
    
    if os.path.exists(config.raw_data_dir):
        existing_files = os.listdir(config.raw_data_dir)
        for required_file in required_files:
            if required_file not in existing_files:
                errors.append(f"Required PCAP file not found: {required_file}")
    
    # Validate ratio parameters
    if not (0.0 <= config.val_ratio <= 1.0):
        errors.append(f"val_ratio must be between 0.0 and 1.0, got: {config.val_ratio}")
    
    if not (0.0 <= config.test_ratio <= 1.0):
        errors.append(f"test_ratio must be between 0.0 and 1.0, got: {config.test_ratio}")
    
    if config.val_ratio + config.test_ratio > 1.0:
        errors.append(f"val_ratio + test_ratio cannot exceed 1.0, got: {config.val_ratio + config.test_ratio}")
    
    # Validate timeout parameters
    if config.idle_timeout <= 0:
        errors.append(f"idle_timeout must be positive, got: {config.idle_timeout}")
    
    if config.active_timeout <= 0:
        errors.append(f"active_timeout must be positive, got: {config.active_timeout}")
    
    return errors


def create_output_directory(config: ProcessingConfig) -> None:
    """
    Create output directory if it doesn't exist.
    
    Args:
        config: Configuration object containing output directory path
        
    Raises:
        OSError: If directory cannot be created
    """
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)


def print_config_summary(config: ProcessingConfig, verbose: bool = False) -> None:
    """
    Print configuration summary.
    
    Args:
        config: Configuration object to summarize
        verbose: Whether to print detailed configuration
    """
    print("=== CIC-2017 Data Processing Configuration ===")
    print(f"Raw data directory: {config.raw_data_dir}")
    print(f"Output directory: {config.output_dir}")
    
    if verbose:
        print(f"NFStreamer idle timeout: {config.idle_timeout}ms")
        print(f"NFStreamer active timeout: {config.active_timeout}ms")
        print(f"Validation ratio: {config.val_ratio}")
        print(f"Test ratio: {config.test_ratio}")
        print(f"Time column: {config.time_col}")
        print(f"Source column: {config.src_col}")
        print(f"Destination column: {config.dst_col}")
    
    print("=" * 47)
