#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data processing utilities for CIC-2017 dataset preprocessing.

This module contains helper functions for data transformation,
feature engineering, and preprocessing operations.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler


def time_sec_to_hhmm(time):
    """
    Convert timestamp in milliseconds to human-readable format.
    
    Args:
        time: Timestamp in milliseconds
        
    Returns:
        str: Formatted time string "DD:MM:YYYY HH:MM:SS"
    """
    time = datetime.fromtimestamp(time / 1000)
    time_str = time.strftime("%d:%m:%Y %H:%M:%S")
    return time_str


def time_hhmm_to_sec(time):
    """
    Convert human-readable time format to timestamp in milliseconds.
    
    Args:
        time: Time string in format "DD:MM:YYYY HH:MM"
        
    Returns:
        float: Timestamp in milliseconds
    """
    time = datetime.strptime(time, "%d:%m:%Y %H:%M")
    time_in_sec = time.timestamp()
    time_in_ms = time_in_sec * 1000
    return time_in_ms


def sort_time(df, time_col):
    """
    Sort DataFrame by time column.
    
    Args:
        df: Input DataFrame
        time_col: Name of time column to sort by
        
    Returns:
        pd.DataFrame: Sorted DataFrame
    """
    return df.sort_values(time_col, inplace=False)


def diff_dst2src_src2dst_first_packet(df, dst2src='dst2src_first_seen_ms', src2dst='src2dst_first_seen_ms'):
    """
    Calculate time difference between first dst->src and src->dst packets.
    
    Args:
        df: Input DataFrame
        dst2src: Column name for dst->src first seen time
        src2dst: Column name for src->dst first seen time
        
    Returns:
        pd.DataFrame: DataFrame with modified dst2src column containing time differences
    """
    dif_time = df.loc[:, dst2src] - df.loc[:, src2dst]
    dif_time[df.loc[:, dst2src] == 0] = 0
    df.loc[:, dst2src] = dif_time
    return df


def substr_min_time(df, time_col):
    """
    Normalize time column by subtracting minimum time.
    
    Args:
        df: Input DataFrame
        time_col: Name of time column to normalize
        
    Returns:
        pd.DataFrame: DataFrame with normalized time column
    """
    df.loc[:, time_col] = df.loc[:, time_col] - df.loc[:, time_col].min()
    return df


def ip_to_id(df, src, dst):
    """
    Convert IP addresses to numerical IDs.
    
    Args:
        df: Input DataFrame
        src: Source IP column name
        dst: Destination IP column name
        
    Returns:
        pd.DataFrame: DataFrame with IP addresses converted to numerical IDs
    """
    all_ip = pd.concat([df['src_ip'], df['dst_ip']], axis=0)
    id_arr, _ = pd.factorize(all_ip)
    l = int(len(id_arr)/2)
    df['src_ip'] = id_arr[:l]
    df['dst_ip'] = id_arr[l:]
    return df


def col_factorize(df, col_to_factorize='application_category_name'):
    """
    Convert categorical column to numerical codes.
    
    Args:
        df: Input DataFrame
        col_to_factorize: Column name to factorize
        
    Returns:
        tuple: (DataFrame with factorized column, unique values, codes)
    """
    codes, uniques = pd.factorize(df.loc[:, col_to_factorize])
    df.loc[:, col_to_factorize] = codes
    return df, uniques, codes


def col_drop(df, col_drop_list):
    """
    Drop specified columns from DataFrame.
    
    Args:
        df: Input DataFrame
        col_drop_list: List of column names to drop
        
    Returns:
        pd.DataFrame: DataFrame with specified columns removed
    """
    return df.drop(col_drop_list, axis=1)


def col_scaler(df, col_to_scale, val_ratio: float = 0.15, test_ratio: float = 0.15):
    """
    Scale numerical columns using MinMaxScaler fitted on temporal train split.
    
    Args:
        df: Input DataFrame
        col_to_scale: List of column names to scale
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        
    Returns:
        pd.DataFrame: DataFrame with scaled columns
    """
    val_time, test_time = np.quantile(
        df.loc[:, 'bidirectional_first_seen_ms'],
        [1. - val_ratio - test_ratio, 1. - test_ratio])

    train_mask = df.loc[:, 'bidirectional_first_seen_ms'] <= val_time
    df_train = df.loc[train_mask, :].copy()
     
    val_mask = (df.loc[:, 'bidirectional_first_seen_ms'] > val_time) & (df.loc[:, 'bidirectional_first_seen_ms'] <= test_time)
    df_val = df.loc[val_mask, :].copy()
     
    test_mask = df.loc[:, 'bidirectional_first_seen_ms'] > test_time
    df_test = df.loc[test_mask, :].copy()
    
    scaler = MinMaxScaler()
    scaler.fit(df_train.loc[:, col_to_scale])
    
    df_train.loc[:, col_to_scale] = scaler.transform(df_train.loc[:, col_to_scale])
    
    if df_val.shape[0] > 0:
        df_val.loc[:, col_to_scale] = scaler.transform(df_val.loc[:, col_to_scale])
        
    if df_test.shape[0] > 0:
        df_test.loc[:, col_to_scale] = scaler.transform(df_test.loc[:, col_to_scale])
        
    return pd.concat([df_train, df_val, df_test], axis=0)


def get_filenames(data_dir_path):
    """
    Get list of PCAP filenames and corresponding CSV names.
    
    Args:
        data_dir_path: Path to directory containing PCAP files
        
    Returns:
        tuple: (list of PCAP filenames, list of corresponding CSV names)
    """
    filenames = [f for f in os.listdir(data_dir_path) if f.endswith('.pcap')]
    print("Found PCAP files:")
    for filename in filenames:
        print(f'  {os.path.join(data_dir_path, filename)}')
    
    names_for_csv = [filename[:-5].lower().replace("-", "_") for filename in filenames]
    return filenames, names_for_csv