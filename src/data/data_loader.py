#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loader module for TGN-SVDD experiments.

This module provides a clean interface for loading temporal graph data
and creating data loaders with proper train/validation/test splits.
"""

from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass

import torch
import numpy as np
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn.models.tgn import LastNeighborLoader

import sys
import os

# Add src directory to path for imports
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Now import everything normally
from config.experiment_config import TGNSVDDConfig
from utils.temporal_data_csv import temporal_data_from_csv


@dataclass
class DataSplit:
    """Container for train/validation/test data splits."""
    
    train_data: TemporalData
    val_data: TemporalData
    test_data: TemporalData
    
    # Data loaders
    train_loader: TemporalDataLoader
    val_loader: TemporalDataLoader
    test_loader: TemporalDataLoader
    
    # Additional metadata
    num_nodes: int
    num_events: int
    y_names: List[str]
    label_distribution: Dict[str, int]


@dataclass 
class DataStatistics:
    """Container for data statistics and metadata."""
    
    total_nodes: int
    total_events: int
    feature_dim: int
    msg_dim: int
    
    # Label statistics
    overall_distribution: Dict[int, int]
    train_distribution: Dict[int, int]
    val_distribution: Dict[int, int]
    test_distribution: Dict[int, int]
    
    # Node statistics
    min_dst_idx: int
    max_dst_idx: int


class TGNDataLoader:
    """
    Data loader for TGN-SVDD experiments.
    
    This class manages the complete data loading pipeline including:
    - Loading temporal data from CSV files
    - Creating train/validation/test splits
    - Setting up data loaders and neighbor loaders
    - Computing data statistics
    """
    
    def __init__(self, config: TGNSVDDConfig, device: torch.device):
        """
        Initialize the data loader.
        
        Args:
            config: Configuration object containing data parameters
            device: PyTorch device for tensor operations
        """
        self.config = config
        self.device = device
        
        # Initialize state
        self.data = None
        self.data_split = None
        self.statistics = None
        self.neighbor_loader = None
        
    def load_day_data(self, day: str) -> DataSplit:
        """
        Load and prepare data for a specific day.
        
        Args:
            day: Day identifier (e.g., 'monday_friday_workinghours')
            
        Returns:
            DataSplit object containing all prepared data and loaders
        """
        # Load raw temporal data
        self.data, y_names = temporal_data_from_csv(
            data_name=day,
            data_split=self.config.data_split
        )
        self.data.to(self.device)
        
        # Compute basic statistics
        self._compute_statistics()
        
        # Create data splits
        train_data, val_data, test_data = self.data.train_val_test_split(
            val_ratio=self.config.val_ratio,
            test_ratio=self.config.test_ratio
        )
        
        # Create data loaders
        train_loader = TemporalDataLoader(train_data, batch_size=self.config.batch_size)
        val_loader = TemporalDataLoader(val_data, batch_size=self.config.batch_size)
        test_loader = TemporalDataLoader(test_data, batch_size=self.config.batch_size)
        
        # Create neighbor loader
        self.neighbor_loader = LastNeighborLoader(
            self.data.num_nodes,
            size=self.config.neighbor_size,
            device=self.device
        )
        
        # Create label distribution summary
        label_distribution = self._compute_label_distributions(
            train_data, val_data, test_data
        )
        
        # Create data split object
        self.data_split = DataSplit(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_nodes=self.data.num_nodes,
            num_events=self.data.num_events,
            y_names=y_names,
            label_distribution=label_distribution
        )
        
        return self.data_split
    
    def get_neighbor_loader(self) -> LastNeighborLoader:
        """
        Get the neighbor loader for the current data.
        
        Returns:
            LastNeighborLoader instance
        """
        if self.neighbor_loader is None:
            raise RuntimeError("Data not loaded yet. Call load_day_data() first.")
        return self.neighbor_loader
    
    def get_data_statistics(self) -> DataStatistics:
        """
        Get comprehensive data statistics.
        
        Returns:
            DataStatistics object with detailed statistics
        """
        if self.statistics is None:
            raise RuntimeError("Data not loaded yet. Call load_day_data() first.")
        return self.statistics
    
    def _compute_statistics(self) -> None:
        """Compute comprehensive data statistics."""
        # Basic data dimensions
        total_nodes = self.data.num_nodes
        total_events = self.data.num_events
        feature_dim = self.data.x.size(-1) if hasattr(self.data, 'x') and self.data.x is not None else 0
        msg_dim = self.data.msg.size(-1)
        
        # Overall label distribution
        unique_labels, counts = torch.unique(self.data.y, return_counts=True)
        overall_distribution = dict(zip(unique_labels.cpu().numpy(), counts.cpu().numpy()))
        
        # Node range for negative sampling
        min_dst_idx = int(self.data.dst.min())
        max_dst_idx = int(self.data.dst.max())
        
        self.statistics = DataStatistics(
            total_nodes=total_nodes,
            total_events=total_events,
            feature_dim=feature_dim,
            msg_dim=msg_dim,
            overall_distribution=overall_distribution,
            train_distribution={},  # Will be filled in _compute_label_distributions
            val_distribution={},
            test_distribution={},
            min_dst_idx=min_dst_idx,
            max_dst_idx=max_dst_idx
        )
    
    def _compute_label_distributions(self, train_data: TemporalData, 
                                   val_data: TemporalData, 
                                   test_data: TemporalData) -> Dict[str, Dict[int, int]]:
        """
        Compute label distributions for each split.
        
        Args:
            train_data: Training data
            val_data: Validation data  
            test_data: Test data
            
        Returns:
            Dictionary mapping split names to label distributions
        """
        distributions = {}
        
        for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
            unique_labels, counts = torch.unique(split_data.y, return_counts=True)
            distribution = dict(zip(unique_labels.cpu().numpy(), counts.cpu().numpy()))
            distributions[split_name] = distribution
            
            # Also update statistics object
            if split_name == "train":
                self.statistics.train_distribution = distribution
            elif split_name == "val":
                self.statistics.val_distribution = distribution
            elif split_name == "test":
                self.statistics.test_distribution = distribution
        
        return distributions
    
    def print_data_summary(self, day: str) -> None:
        """
        Print a comprehensive summary of the loaded data.
        
        Args:
            day: Day identifier for the loaded data
        """
        if self.data_split is None or self.statistics is None:
            print("❌ No data loaded yet. Call load_day_data() first.")
            return
        
        print("=" * 60)
        print(f"📊 Data Summary: {day}")
        print("=" * 60)
        
        # Basic statistics
        print(f"Total nodes: {self.statistics.total_nodes:,}")
        print(f"Total events: {self.statistics.total_events:,}")
        print(f"Message dimension: {self.statistics.msg_dim}")
        print(f"Feature dimension: {self.statistics.feature_dim}")
        print(f"Node range: {self.statistics.min_dst_idx} - {self.statistics.max_dst_idx}")
        
        # Label distributions
        print("\n📈 Label Distributions:")
        print(f"Overall: {self.statistics.overall_distribution}")
        print(f"Train:   {self.statistics.train_distribution}")
        print(f"Val:     {self.statistics.val_distribution}")
        print(f"Test:    {self.statistics.test_distribution}")
        
        # Split sizes
        print("\n📋 Split Sizes:")
        print(f"Train: {self.data_split.train_data.num_events:,} events")
        print(f"Val:   {self.data_split.val_data.num_events:,} events")
        print(f"Test:  {self.data_split.test_data.num_events:,} events")
        
        # Batch information
        print(f"\n🔄 Batch Information:")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Train batches: {len(self.data_split.train_loader)}")
        print(f"Val batches:   {len(self.data_split.val_loader)}")
        print(f"Test batches:  {len(self.data_split.test_loader)}")
        
        print("=" * 60)


def create_data_loader(config: TGNSVDDConfig, device: torch.device) -> TGNDataLoader:
    """
    Factory function to create a TGN data loader.
    
    Args:
        config: Configuration object
        device: PyTorch device
        
    Returns:
        Configured TGNDataLoader instance
    """
    return TGNDataLoader(config, device)


# Convenience functions for backward compatibility
def load_day_data(day: str, config: TGNSVDDConfig, device: torch.device) -> Tuple[DataSplit, TGNDataLoader]:
    """
    Convenience function to load data for a specific day.
    
    Args:
        day: Day identifier
        config: Configuration object
        device: PyTorch device
        
    Returns:
        Tuple of (DataSplit, TGNDataLoader)
    """
    data_loader = create_data_loader(config, device)
    data_split = data_loader.load_day_data(day)
    return data_split, data_loader


if __name__ == "__main__":
    """Test the data loader functionality."""
    print("🧪 Testing TGN Data Loader")
    print("-" * 40)
    
    # This would require the actual data and dependencies
    # For now, just test that the classes can be instantiated
    try:
        # Import config with fallback
        try:
            from ..config.experiment_config import TGNSVDDConfig
        except ImportError:
            from config.experiment_config import TGNSVDDConfig
        
        config = TGNSVDDConfig()
        print("✅ TGNSVDDConfig created successfully")
        
        # Note: Can't test full functionality without PyTorch environment
        print("✅ Data loader module structure is valid")
        print("🎉 Data loader test completed")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
