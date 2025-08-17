#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CIC-2017 Dataset Preprocessing for TGN-SVDD

This script processes raw CIC-2017 PCAP files into CSV format suitable for
Temporal Graph Network (TGN) based anomaly detection experiments.

Original created on Wed Nov 10 17:47:31 2021
@author: aleksei

Refactored for better maintainability and configurability.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings

import nfstream
from nfstream import NFStreamer, NFPlugin

# Suppress pandas DtypeWarning from NFStream's internal CSV processing
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)

# Import our configuration system
from config import parse_arguments, validate_config, create_output_directory, print_config_summary
# Import utility functions
from utils import (
    time_sec_to_hhmm, time_hhmm_to_sec, sort_time, diff_dst2src_src2dst_first_packet,
    substr_min_time, ip_to_id, col_factorize, col_drop, col_scaler, get_filenames
)


print(f"NFStream version: {nfstream.__version__}")


# Attack labeling rules for CIC-2017 dataset
ATTACK_RULES = {
    'Tuesday-WorkingHours.pcap': [
        ('172.16.0.1', '192.168.10.50', '04:07:2017 13:53', '04:07:2017 22:01', 'SSH-Patator/FTP-Patator')
    ],
    
    'Wednesday-WorkingHours.pcap': [
        ('172.16.0.1', '192.168.10.51', '05:07:2017 20:11', '05:07:2017 20:33', 'Heartbleed'),
        ('172.16.0.1', '192.168.10.50', '05:07:2017 14:00', '05:07:2017 19:26', 'DoS_slowloris'),
        ('172.16.0.1', '192.168.10.50', '05:07:2017 15:14', '05:07:2017 15:38', 'DoS_Slowhttptest'),
        ('172.16.0.1', '192.168.10.50', '05:07:2017 15:42', '05:07:2017 16:08', 'DoS_Hulk'),
        ('172.16.0.1', '192.168.10.50', '05:07:2017 16:10', '05:07:2017 16:20', 'DoS_GoldenEye')
    ],
    
    'Thursday-WorkingHours.pcap': [
        ('172.16.0.1', '192.168.10.50', '06:07:2017 15:14', '06:07:2017 15:35', 'Web_Attack_XSS'),
        ('172.16.0.1', '192.168.10.50', '06:07:2017 15:39', '06:07:2017 15:43', 'Web_Attack_Sql_Injection'),
        ('172.16.0.1', '192.168.10.50', '06:07:2017 14:15', '06:07:2017 15:01', 'Web_Attack_brute_forse'),
        ('192.168.10.8', '205.174.165.73', '06:07:2017 19:18', '06:07:2017 20:46', 'INFILTRATION ')
    ],
    
    'Friday-WorkingHours.pcap': [
        ('192.168.10.5', '205.174.165.73', '07:07:2017 14:30', '07:07:2017 18:05', 'Bot'),
        ('192.168.10.8', '205.174.165.73', '07:07:2017 14:30', '07:07:2017 18:05', 'Bot'),
        ('192.168.10.9', '205.174.165.73', '07:07:2017 14:30', '07:07:2017 18:05', 'Bot'),
        ('192.168.10.14', '205.174.165.73', '07:07:2017 14:30', '07:07:2017 18:05', 'Bot'),
        ('192.168.10.15', '205.174.165.73', '07:07:2017 14:30', '07:07:2017 18:05', 'Bot'),
        ('205.174.165.73', '192.168.10.5', '07:07:2017 14:30', '07:07:2017 18:05', 'Bot'),
        ('205.174.165.73', '192.168.10.8', '07:07:2017 14:30', '07:07:2017 18:05', 'Bot'),
        ('205.174.165.73', '192.168.10.9', '07:07:2017 14:30', '07:07:2017 18:05', 'Bot'),
        ('205.174.165.73', '192.168.10.14', '07:07:2017 14:30', '07:07:2017 18:05', 'Bot'),
        ('205.174.165.73', '192.168.10.15', '07:07:2017 14:30', '07:07:2017 18:05', 'Bot'),
        ('172.16.0.1', '192.168.10.50', '07:07:2017 18:04', '07:07:2017 20:24', 'PortScan'),
        ('172.16.0.1', '192.168.10.50', '07:07:2017 20:55', '07:07:2017 21:18', 'DDoS')
    ]
}


def label_attacks(df, pcap_file_name, verbose=False):
    """
    Label network flows with attack information based on CIC-2017 ground truth.
    
    Args:
        df: DataFrame with network flows
        pcap_file_name: Name of the PCAP file being processed
        verbose: Whether to print detailed labeling information
        
    Returns:
        pd.DataFrame: DataFrame with added 'label' and 'attack' columns
    """
    # Initialize label columns
    df['label'] = 0
    df['attack'] = 'normal'
    
    if pcap_file_name not in ATTACK_RULES:
        if verbose:
            print(f"No attack rules defined for {pcap_file_name}")
        return df
    
    # Convert attack time rules to integer timestamps
    attack_rules_ms = []
    for rule in ATTACK_RULES[pcap_file_name]:
        src_ip, dst_ip, start_time, end_time, attack_type = rule
        start_ms = time_hhmm_to_sec(start_time)
        end_ms = time_hhmm_to_sec(end_time)
        attack_rules_ms.append([src_ip, dst_ip, start_ms, end_ms, attack_type])
    
    # Apply labeling rules
    for attack in attack_rules_ms:
        src_ip, dst_ip, start_ms, end_ms, attack_type = attack
        
        mask = ((df.src_ip.isin([src_ip])) & (df.dst_ip.isin([dst_ip])) &
                (start_ms <= df['bidirectional_first_seen_ms']) & 
                (df['bidirectional_first_seen_ms'] <= end_ms))
        
        labeled_count = mask.sum()
        df.loc[mask, 'label'] = 1
        df.loc[mask, 'attack'] = attack_type
        
        if verbose:
            print(f"  Labeled {labeled_count} flows as '{attack_type}' ({src_ip} -> {dst_ip})")
    
    attack_count = (df['label'] == 1).sum()
    normal_count = (df['label'] == 0).sum()
    
    if verbose:
        print(f"  Total: {normal_count} normal, {attack_count} attack flows")
        
    return df


def pcap_to_flow(data_dir_path, pcap_file_name, idle_timeout=1, active_timeout=1, verbose=False):
    """
    Convert PCAP file to network flow DataFrame with attack labeling.
    
    Args:
        data_dir_path: Directory containing PCAP files
        pcap_file_name: Name of PCAP file to process  
        idle_timeout: NFStreamer idle timeout in ms
        active_timeout: NFStreamer active timeout in ms
        verbose: Whether to print detailed processing information
        
    Returns:
        pd.DataFrame: Network flows with features and attack labels
    """
    if verbose:
        print(f"  Converting {pcap_file_name} to flows...")
    
    file_path = os.path.join(data_dir_path, pcap_file_name)
    
    my_streamer = NFStreamer(
        source=file_path,
        decode_tunnels=True,
        bpf_filter=None,
        promiscuous_mode=True,
        snapshot_length=1536,
        idle_timeout=idle_timeout,
        active_timeout=active_timeout,
        accounting_mode=0,
        udps=None,
        n_dissections=20,
        statistical_analysis=True,
        splt_analysis=0,
        n_meters=0,
        performance_report=0
    )
    
    df = my_streamer.to_pandas()
    
    # Label attacks based on ground truth rules
    df = label_attacks(df, pcap_file_name, verbose)
    
    return df


def pcap_to_flow_old(data_dir_path, pcap_file_name, idle_timeout=1, active_timeout=1):
    # day_n = 2
    # print(filenames[day_n])
    print(pcap_file_name)
    
    
    my_streamer = NFStreamer(source=f'{data_dir_path}{pcap_file_name}',
                             decode_tunnels=True,
                             bpf_filter=None,
                             promiscuous_mode=True,
                             snapshot_length=1536,
                             idle_timeout=idle_timeout,
                             active_timeout=active_timeout,
                             accounting_mode=0,
                             udps=None,
                             n_dissections=20,
                             statistical_analysis=True,
                             splt_analysis=0,
                             n_meters=0,
                             performance_report=0)
    
    df = my_streamer.to_pandas()
    # Create label columns
    df['label'] = 0
    df['attack'] = 'normal'
    
    
    # the rules to label the flows
    attack_week = { 'Tuesday-WorkingHours.pcap' :
                      [('172.16.0.1', '192.168.10.50', '04:07:2017 13:53', '04:07:2017 22:01', 'SSH-Patator/FTP-Patator')],
                     
                      'Wednesday-WorkingHours.pcap':
                                   
                      [('172.16.0.1', '192.168.10.51', '05:07:2017 20:11', '05:07:2017 20:33', 'Heartbleed'),
                     
                      ('172.16.0.1', '192.168.10.50', '05:07:2017 14:00', '05:07:2017 19:26', 'DoS_slowloris'),
                      ('172.16.0.1', '192.168.10.50', '05:07:2017 15:14', '05:07:2017 15:38', 'DoS_Slowhttptest'),
                      ('172.16.0.1', '192.168.10.50', '05:07:2017 15:42', '05:07:2017 16:08', 'DoS_Hulk'),
                      ('172.16.0.1', '192.168.10.50', '05:07:2017 16:10', '05:07:2017 16:20', 'DoS_GoldenEye')
                      ],
                    
                    'Thursday-WorkingHours.pcap':
                    [('172.16.0.1', '192.168.10.50', '06:07:2017 15:14', '06:07:2017 15:35', 'Web_Attack_XSS'),
                      ('172.16.0.1', '192.168.10.50', '06:07:2017 15:39', '06:07:2017 15:43', 'Web_Attack_Sql_Injection'),
                      ('172.16.0.1', '192.168.10.50', '06:07:2017 14:15', '06:07:2017 15:01', 'Web_Attack_brute_forse'),
                     
                      ('192.168.10.8', '205.174.165.73', '06:07:2017 19:18', '06:07:2017 20:46', 'INFILTRATION ')
                      ],
                    'Friday-WorkingHours.pcap':
                    [('192.168.10.5', '205.174.165.73', '07:07:2017 14:30', '07:07:2017 18:05', 'Bot'),
                      ('192.168.10.8', '205.174.165.73', '07:07:2017 14:30', '07:07:2017 18:05', 'Bot'),
                      ('192.168.10.9', '205.174.165.73', '07:07:2017 14:30', '07:07:2017 18:05', 'Bot'),
                      ('192.168.10.14', '205.174.165.73', '07:07:2017 14:30', '07:07:2017 18:05', 'Bot'),
                      ('192.168.10.15', '205.174.165.73', '07:07:2017 14:30', '07:07:2017 18:05', 'Bot'),
                      ('205.174.165.73', '192.168.10.5', '07:07:2017 14:30', '07:07:2017 18:05', 'Bot'),
                      ('205.174.165.73', '192.168.10.8', '07:07:2017 14:30', '07:07:2017 18:05', 'Bot'),
                      ('205.174.165.73', '192.168.10.9', '07:07:2017 14:30', '07:07:2017 18:05', 'Bot'),
                      ('205.174.165.73', '192.168.10.14', '07:07:2017 14:30', '07:07:2017 18:05', 'Bot'),
                      ('205.174.165.73', '192.168.10.15', '07:07:2017 14:30', '07:07:2017 18:05', 'Bot'),
                                      
                     
                      ('172.16.0.1', '192.168.10.50', '07:07:2017 18:04', '07:07:2017 20:24', 'PortScan'),
                      ('172.16.0.1', '192.168.10.50', '07:07:2017 20:55', '07:07:2017 21:18', 'DDoS')
                      ]
                    }
    
    
       
    
    
    if pcap_file_name in attack_week:
        #
        
        # transform time from date str to ms int
        attakc_int_time ={}
        
        for att_day in attack_week:
            
            
            temp_tup = []
            
            for x in attack_week[att_day]:
                temp_tup.append([x[0], x[1], time_hhmm_to_sec(x[2]), time_hhmm_to_sec(x[3]), x[4]])
                
            attakc_int_time[att_day] = temp_tup
            
        # attakc_int_time
        
        
        
        
        attacks = attakc_int_time[pcap_file_name]
        
        #
        for attack in attacks:
            
            print(attack)
            mask = ((df.src_ip.isin([attack[0]])) & (df.dst_ip.isin([attack[1]])) &
                      (attack[2] <= df['bidirectional_first_seen_ms']) & 
                      (df['bidirectional_first_seen_ms'] <= attack[3]))
            
            print('mask unique', mask.unique())
            print('mask # of values', mask.value_counts())
            
            df.loc[mask, 'label'] = 1
            df.loc[mask, 'attack'] = attack[4]
            print(df['label'].unique())
            print(df['label'].value_counts())
            print(df['attack'].unique())
            print(df['attack'].value_counts())
            
        
        print(df['label'].unique())
        print(df['label'].value_counts())
        print(df['attack'].unique())
        print(df['attack'].value_counts())
    
    return df

    
      












############################ main func ##################

# here specify some parametrs and process data PCAP -> NFlows





time_col='bidirectional_first_seen_ms'
src='src_ip'
dst='dst_ip'
# src_attack, dst_attack = ['172.16.0.1'], ['192.168.10.50']
col_to_factorize = 'application_category_name'# 3 times in this case

col_drop_list = list(['id', 'expiration_id', 'src_mac', 'src_oui', 'src_port', 'dst_mac', 'dst_oui', 'dst_port', 'protocol', 'vlan_id', 'application_name',
 'application_confidence', 'requested_server_name', 'client_fingerprint', 'server_fingerprint', 'user_agent',
 'src2dst_first_seen_ms', 'application_is_guessed', 'bidirectional_last_seen_ms', 'content_type', 'src2dst_last_seen_ms', 'dst2src_last_seen_ms'])

col_to_scale = list(['bidirectional_duration_ms', 'bidirectional_packets', 'bidirectional_bytes', 'src2dst_duration_ms', 'src2dst_packets', 'src2dst_bytes',
 'dst2src_first_seen_ms', 'dst2src_duration_ms', 'dst2src_packets', 'dst2src_bytes', 'bidirectional_min_ps', 'bidirectional_mean_ps', 'bidirectional_stddev_ps',
 'bidirectional_max_ps', 'src2dst_min_ps', 'src2dst_mean_ps', 'src2dst_stddev_ps', 'src2dst_max_ps', 'dst2src_min_ps', 'dst2src_mean_ps', 'dst2src_stddev_ps',
 'dst2src_max_ps', 'bidirectional_min_piat_ms', 'bidirectional_mean_piat_ms', 'bidirectional_stddev_piat_ms', 'bidirectional_max_piat_ms', 'src2dst_min_piat_ms',
 'src2dst_mean_piat_ms', 'src2dst_stddev_piat_ms', 'src2dst_max_piat_ms', 'dst2src_min_piat_ms', 'dst2src_mean_piat_ms', 'dst2src_stddev_piat_ms', 'dst2src_max_piat_ms',
 'bidirectional_syn_packets', 'bidirectional_cwr_packets', 'bidirectional_ece_packets', 'bidirectional_urg_packets', 'bidirectional_ack_packets',
 'bidirectional_psh_packets', 'bidirectional_rst_packets', 'bidirectional_fin_packets', 'src2dst_syn_packets', 'src2dst_cwr_packets', 'src2dst_ece_packets',
 'src2dst_urg_packets', 'src2dst_ack_packets', 'src2dst_psh_packets', 'src2dst_rst_packets', 'src2dst_fin_packets', 'dst2src_syn_packets', 'dst2src_cwr_packets',
 'dst2src_ece_packets', 'dst2src_urg_packets', 'dst2src_ack_packets', 'dst2src_psh_packets', 'dst2src_rst_packets',
 'dst2src_fin_packets'])

idle_timeout=1000
active_timeout=3000 

val_ratio = 0
test_ratio = 0


def data_process(df, src, dst, col_drop_list, col_to_scale, val_ratio, test_ratio, src_attack=None, dst_attack=None, time_col='bidirectional_first_seen_ms'):
    """
    Orchestrates all feature engineering and transformation steps using utility functions.
    """
    # Sort by time
    df = sort_time(df, time_col=time_col)
    # Feature: difference between first dst2src and src2dst packet
    df = diff_dst2src_src2dst_first_packet(df)
    # Subtract minimum time
    df = substr_min_time(df, time_col=time_col)
    # Factorize IPs
    df = ip_to_id(df, src=src, dst=dst)
    # Factorize categorical columns
    for col in ['application_category_name', 'tunnel_id', 'ip_version']:
        df, _, _ = col_factorize(df, col_to_factorize=col)
    # Drop unused columns
    df = col_drop(df, col_drop_list=col_drop_list)
    # Scale columns
    df = col_scaler(df, col_to_scale=col_to_scale, val_ratio=val_ratio, test_ratio=test_ratio)
    return df






def process_single_file(config, pcap_file, verbose=False):
    """
    Process a single PCAP file with proper error handling and progress reporting.
    
    Args:
        config: ProcessingConfig object with all parameters
        pcap_file: Name of PCAP file to process
        verbose: Whether to print detailed progress information
        
    Returns:
        pd.DataFrame: Processed DataFrame with network flows and attack labels
    """
    if verbose:
        print(f"  Converting {pcap_file} to flows...")
    
    try:
        df = pcap_to_flow(
            data_dir_path=config.raw_data_dir,
            pcap_file_name=pcap_file,
            idle_timeout=config.idle_timeout,
            active_timeout=config.active_timeout,
            verbose=verbose
        )
        
        if verbose:
            print(f"  Generated {len(df)} network flows")
            print(f"  Attack flows: {(df['label'] == 1).sum()}")
            print(f"  Normal flows: {(df['label'] == 0).sum()}")
        
        return df
        
    except Exception as e:
        print(f"ERROR processing {pcap_file}: {str(e)}")
        raise


def main():
    """Main processing function with improved organization and progress reporting."""
    # Parse arguments and get configuration
    config, dry_run, verbose = parse_arguments()
    
    # Print configuration summary
    print_config_summary(config, verbose)
    
    # Validate configuration
    errors = validate_config(config)
    if errors:
        print("Configuration validation failed:")
        for error in errors:
            print(f"  ERROR: {error}")
        return 1
    
    # Create output directory
    try:
        create_output_directory(config)
        if verbose:
            print(f"Output directory ready: {config.output_dir}")
    except OSError as e:
        print(f"ERROR: Could not create output directory: {e}")
        return 1
    
    if dry_run:
        print("DRY RUN: Configuration is valid, exiting without processing.")
        return 0
    
    # Start processing
    print("Starting CIC-2017 data processing...")
    print(f"Processing files from: {config.raw_data_dir}")
    print(f"Output directory: {config.output_dir}")
    
    # Get filenames from raw data directory  
    filenames, names_for_csv = get_filenames(config.raw_data_dir)
    if 'Monday-WorkingHours.pcap' not in filenames:
        print("ERROR: 'Monday-WorkingHours.pcap' not found in raw data directory.")
        return 1
    
    # Remove Monday from attack files list (it will be processed separately as normal baseline)
    filenames.remove('Monday-WorkingHours.pcap')
    names_for_csv = [filename[:-5].lower().replace("-", "_") for filename in filenames]
    
    if verbose:
        print(f"Files to process: {len(filenames)} attack days + Monday baseline")
        print(f"Attack files: {filenames}")
    
    try:
        # Step 1: Process Monday as baseline (normal traffic)
        print(f"\n{'='*60}")
        print("Step 1: Processing Monday baseline (normal traffic)")
        print(f"{'='*60}")
        
        df_monday = process_single_file(
            config=config,
            pcap_file='Monday-WorkingHours.pcap',
            verbose=verbose
        )
        max_time_monday = df_monday[config.time_col].max()
        
        if verbose:
            print(f"Monday baseline processed: {len(df_monday)} flows")
            print(f"Maximum timestamp: {max_time_monday}")
        
        # Step 2: Process each attack day and combine with Monday
        print(f"\n{'='*60}")
        print("Step 2: Processing attack days and creating datasets")
        print(f"{'='*60}")
        
        for i, (pcap_file_name, save_csv_name) in enumerate(zip(filenames, names_for_csv), 1):
            print(f"\n[{i}/{len(filenames)}] Processing {pcap_file_name}...")
            
            # Process attack day
            df_attack = process_single_file(
                config=config,
                pcap_file=pcap_file_name,
                verbose=verbose
            )
            
            # Adjust attack day timestamps to continue from Monday
            min_time_attack = df_attack[config.time_col].min()
            df_attack[config.time_col] = df_attack[config.time_col] - min_time_attack + max_time_monday
            
            if verbose:
                print(f"  Timestamp adjustment: attack day starts at {max_time_monday}")
            
            # Combine Monday + attack day
            df_combined = pd.concat([df_monday, df_attack], ignore_index=True)
            
            if verbose:
                print(f"  Combined dataset: {len(df_combined)} flows")
                print(f"    Normal flows: {(df_combined['label'] == 0).sum()}")
                print(f"    Attack flows: {(df_combined['label'] == 1).sum()}")
            
            # Apply final feature engineering
            if verbose:
                print("  Applying final feature engineering...")
            
            df_processed = data_process(
                df=df_combined,
                src=config.src_col,
                dst=config.dst_col,
                col_drop_list=config.col_drop_list,
                col_to_scale=config.col_to_scale,
                val_ratio=config.val_ratio,
                test_ratio=config.test_ratio,
                time_col=config.time_col
            )
            
            # Reorder columns for consistency
            expected_columns = [
                'src_ip', 'dst_ip', 'bidirectional_first_seen_ms', 'label', 'attack', 'ip_version', 'tunnel_id',
                'bidirectional_duration_ms', 'bidirectional_packets', 'bidirectional_bytes', 'src2dst_duration_ms', 'src2dst_packets', 'src2dst_bytes',
                'dst2src_first_seen_ms', 'dst2src_duration_ms', 'dst2src_packets', 'dst2src_bytes', 'bidirectional_min_ps', 'bidirectional_mean_ps', 'bidirectional_stddev_ps',
                'bidirectional_max_ps', 'src2dst_min_ps', 'src2dst_mean_ps', 'src2dst_stddev_ps', 'src2dst_max_ps', 'dst2src_min_ps', 'dst2src_mean_ps', 'dst2src_stddev_ps',
                'dst2src_max_ps', 'bidirectional_min_piat_ms', 'bidirectional_mean_piat_ms', 'bidirectional_stddev_piat_ms', 'bidirectional_max_piat_ms', 'src2dst_min_piat_ms',
                'src2dst_mean_piat_ms', 'src2dst_stddev_piat_ms', 'src2dst_max_piat_ms', 'dst2src_min_piat_ms', 'dst2src_mean_piat_ms', 'dst2src_stddev_piat_ms', 'dst2src_max_piat_ms',
                'bidirectional_syn_packets', 'bidirectional_cwr_packets', 'bidirectional_ece_packets', 'bidirectional_urg_packets', 'bidirectional_ack_packets',
                'bidirectional_psh_packets', 'bidirectional_rst_packets', 'bidirectional_fin_packets', 'src2dst_syn_packets', 'src2dst_cwr_packets', 'src2dst_ece_packets',
                'src2dst_urg_packets', 'src2dst_ack_packets', 'src2dst_psh_packets', 'src2dst_rst_packets', 'src2dst_fin_packets', 'dst2src_syn_packets', 'dst2src_cwr_packets',
                'dst2src_ece_packets', 'dst2src_urg_packets', 'dst2src_ack_packets', 'dst2src_psh_packets', 'dst2src_rst_packets', 'dst2src_fin_packets', 'application_category_name'
            ]
            
            df_processed = df_processed.reindex(columns=expected_columns)
            
            # Save processed dataset
            output_filename = f'monday_{save_csv_name}.csv'
            save_path = os.path.join(config.output_dir, output_filename)
            df_processed.to_csv(save_path, index=False)
            
            print(f"  ✓ Saved: {output_filename} ({len(df_processed)} flows)")
            
            if verbose:
                print(f"    Final shape: {df_processed.shape}")
                print(f"    Attack types: {df_processed['attack'].unique()}")
        
        print(f"\n{'='*60}")
        print("✓ Processing completed successfully!")
        print(f"✓ Generated {len(filenames)} datasets in {config.output_dir}")
        print(f"{'='*60}")
        return 0
        
    except Exception as e:
        print(f"\nERROR during processing: {str(e)}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)










