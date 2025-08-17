#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TGN-SVDD Experiment Orchestrator (Refactored).

This module contains the main experiment class that orchestrates the complete
TGN-SVDD pipeline using modular components for data loading, model creation,
training, and evaluation.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastNeighborLoader,
    MeanAggregator
)

import sys
import os

# Add src directory to path for imports
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Now import everything normally
from config.experiment_config import TGNSVDDConfig
from utils.loger import setup_logger
from utils.plot_functions import plot_nodes_color_kde_precentil_3
from utils.temporal_data_csv import temporal_data_from_csv
from models.models import GraphAttentionEmbedding, DeepSVDD


class TGNSVDDExperiment:
    """
    Main experiment orchestrator for TGN-SVDD intrusion detection.
    
    This class manages the complete pipeline including:
    - Data loading and preprocessing
    - Model initialization and training
    - Evaluation and metrics calculation
    - Results saving and visualization
    """
    
    def __init__(self, config: TGNSVDDConfig):
        """
        Initialize the experiment with configuration.
        
        Args:
            config: Configuration object containing all parameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components (will be set during setup)
        self.data = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.neighbor_loader = None
        self.memory = None
        self.gnn = None
        self.deep_svdd = None
        self.optimizer = None
        self.criterion = None
        self.assoc = None
        self.logger = None
        self.results_path = None
        
    def setup_experiment(self, day: str) -> None:
        """
        Set up the experiment for a specific day.
        
        Args:
            day: Day identifier (e.g., 'monday_friday_workinghours')
        """
        # Create results directory (resolve relative to repo root if not absolute)
        script_time = datetime.now().strftime('%d%m%Y_%H%M%S')
        repo_root = Path(__file__).resolve().parents[2]
        base_results = Path(self.config.results_dir)
        if not base_results.is_absolute():
            base_results = repo_root / base_results
        self.results_path = str(base_results / f"{script_time}_svdd_{day[7:10]}")
        Path(self.results_path).mkdir(parents=True, exist_ok=True)
        
        # Setup logger
        comment = f'TGN-SVDD experiment with CIC {day}'
        setup_info = [
            comment, 
            f'n_epoch={self.config.n_epochs}', 
            f'batch_size={self.config.batch_size}', 
            f'data: {self.config.data_split}_{day}'
        ]
        self.logger = setup_logger(self.results_path, script_time, setup_info)
        
        # Load and prepare data
        self._load_data(day)
        self._create_data_loaders()
        self._initialize_models()
        
        self.logger.info(f"Experiment setup complete for {day}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Data nodes: {self.data.num_nodes}, events: {self.data.num_events}")
        
    def _load_data(self, day: str) -> None:
        """Load and split temporal data."""
        self.data, y_names = temporal_data_from_csv(
            data_name=day, 
            data_split=self.config.data_split
        )
        self.data.to(self.device)
        
        # Log data statistics
        unique_labels, counts = torch.unique(self.data.y, return_counts=True)
        self.logger.info(f"Data labels distribution: {dict(zip(unique_labels.cpu().numpy(), counts.cpu().numpy()))}")
        
        # Split data
        self.train_data, self.val_data, self.test_data = self.data.train_val_test_split(
            val_ratio=self.config.val_ratio, 
            test_ratio=self.config.test_ratio
        )
        
        # Log split statistics
        for split_name, split_data in [("Train", self.train_data), ("Val", self.val_data), ("Test", self.test_data)]:
            unique_labels, counts = torch.unique(split_data.y, return_counts=True)
            self.logger.info(f"{split_name} labels distribution: {dict(zip(unique_labels.cpu().numpy(), counts.cpu().numpy()))}")
    
    def _create_data_loaders(self) -> None:
        """Create data loaders for training, validation, and testing."""
        self.train_loader = TemporalDataLoader(self.train_data, batch_size=self.config.batch_size)
        self.val_loader = TemporalDataLoader(self.val_data, batch_size=self.config.batch_size)
        self.test_loader = TemporalDataLoader(self.test_data, batch_size=self.config.batch_size)
        
        self.neighbor_loader = LastNeighborLoader(
            self.data.num_nodes, 
            size=self.config.neighbor_size, 
            device=self.device
        )
    
    def _initialize_models(self) -> None:
        """Initialize TGN memory, GNN, and Deep SVDD models."""
        # TGN Memory
        self.memory = TGNMemory(
            self.data.num_nodes,
            self.data.msg.size(-1),
            self.config.memory_dim,
            self.config.time_dim,
            message_module=IdentityMessage(
                self.data.msg.size(-1), 
                self.config.memory_dim, 
                self.config.time_dim
            ),
            aggregator_module=MeanAggregator(),
        ).to(self.device)
        
        # Graph Neural Network
        self.gnn = GraphAttentionEmbedding(
            in_channels=self.config.memory_dim,
            out_channels=self.config.embedding_dim,
            msg_dim=self.data.msg.size(-1),
            time_enc=self.memory.time_enc,
        ).to(self.device)
        
        # Deep SVDD
        self.deep_svdd = DeepSVDD(2 * self.config.embedding_dim).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            set(self.memory.parameters()) | set(self.gnn.parameters()) | set(self.deep_svdd.parameters()), 
            lr=self.config.learning_rate
        )
        
        # Loss criterion
        self.criterion = torch.nn.MSELoss()
        
        # Helper vector for node mapping
        self.assoc = torch.empty(self.data.num_nodes, dtype=torch.long, device=self.device)
    
    def train_epoch(self, epoch: int) -> Tuple[float, np.ndarray]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_loss, c_parameter)
        """
        self.memory.train()
        self.gnn.train()
        self.deep_svdd.train()
        
        self.memory.reset_state()
        self.neighbor_loader.reset_state()
        
        total_loss = 0
        
        for batch in tqdm(self.train_loader, total=len(self.train_loader), 
                         desc=f'Epoch {epoch} train progress'):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
            
            # Get neighborhood
            n_id = torch.cat([src, pos_dst]).unique()
            n_id, edge_index, e_id = self.neighbor_loader(n_id)
            self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)
            
            # Get updated memory and embeddings
            z, last_update = self.memory(n_id)
            z = self.gnn(z, last_update, edge_index, 
                        self.data.t[e_id].to(self.device),
                        self.data.msg[e_id].to(self.device))
            
            # Concatenate source and destination embeddings
            emb = torch.cat([z[self.assoc[src]], z[self.assoc[pos_dst]]], dim=1)
            
            # Deep SVDD loss
            distances = self.deep_svdd(emb)
            loss = self.criterion(distances, torch.zeros_like(distances)) + \
                   self.config.regularization_weight * torch.norm(emb, p=2, dim=1).mean()
            
            # Update memory and neighbor loader
            self.memory.update_state(src, pos_dst, t, msg)
            self.neighbor_loader.insert(src, pos_dst)
            
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.memory.detach()
            total_loss += float(loss) * batch.num_events
        
        # Save checkpoint if enabled
        if self.config.save_checkpoints:
            self._save_checkpoint(epoch, total_loss)
        
        c_param = self.deep_svdd.c.detach().cpu().numpy()
        return total_loss / self.train_data.num_events, c_param
    
    @torch.no_grad()
    def evaluate(self, loader: TemporalDataLoader, epoch: int) -> Tuple[float, List[float], np.ndarray]:
        """
        Evaluate the model on a data loader.
        
        Args:
            loader: Data loader for evaluation
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_distance, all_distances, labels)
        """
        self.memory.eval()
        self.gnn.eval()
        self.deep_svdd.eval()
        
        torch.manual_seed(self.config.seed)  # Ensure deterministic sampling
        
        total_distances, label_track = [], []
        
        for batch in tqdm(loader, total=len(loader), 
                         desc=f'Epoch {epoch} eval progress'):
            batch = batch.to(self.device)
            src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg
            
            # Get neighborhood
            n_id = torch.cat([src, pos_dst]).unique()
            n_id, edge_index, e_id = self.neighbor_loader(n_id)
            self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)
            
            # Get embeddings
            z, last_update = self.memory(n_id)
            z = self.gnn(z, last_update, edge_index,
                        self.data.t[e_id].to(self.device),
                        self.data.msg[e_id].to(self.device))
            
            # Concatenate embeddings
            emb = torch.cat([z[self.assoc[src]], z[self.assoc[pos_dst]]], dim=1)
            
            # Get distances
            distances = self.deep_svdd(emb)
            
            # Collect results
            total_distances.extend(distances.cpu().numpy())
            label_track = np.append(label_track, batch.y.detach().cpu().numpy())
            
            # Update state
            self.memory.update_state(src, pos_dst, t, msg)
            self.neighbor_loader.insert(src, pos_dst)
        
        avg_distance = float(torch.tensor(total_distances).mean())
        return avg_distance, total_distances, label_track
    
    def evaluate_and_plot(self, epoch: int) -> Dict[str, float]:
        """
        Run full evaluation and generate plots.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Reset state for evaluation
        self.memory.reset_state()
        self.neighbor_loader.reset_state()
        
        # Evaluate on all splits
        train_dis, train_dis_track, train_label_track = self.evaluate(self.train_loader, epoch)
        val_dis, val_dis_track, val_label_track = self.evaluate(self.val_loader, epoch)
        test_dis, test_dis_track, test_label_track = self.evaluate(self.test_loader, epoch)
        
        self.logger.info(f'Val distance avg.: {val_dis:.4f}')
        self.logger.info(f'Test distance avg.: {test_dis:.4f}')
        
        # Calculate threshold and predictions
        threshold = np.percentile(train_dis_track, self.config.threshold_percentile)
        test_predictions = [1 if dist > threshold else 0 for dist in test_dis_track]
        test_labels_binary = [1 if label == 1 else 0 for label in test_label_track]
        
        # Calculate metrics
        f1 = f1_score(test_labels_binary, test_predictions, pos_label=1)
        
        # Generate plots
        arrays = [train_dis_track, val_dis_track, test_dis_track]
        labels_arr = [train_label_track, val_label_track, test_label_track]
        
        plot_nodes_color_kde_precentil_3(
            arrays=arrays, 
            lables_arr=labels_arr,
            save_path=self.results_path, 
            title=f'tr_val_tst_kde_epoch_{epoch}',
            kde_title=f'f1 score {f1:.4f}', 
            percentile=self.config.threshold_percentile, 
            save=True,
            save_name=f'tr_val_tst_kde_epoch_{epoch}', 
            alpha_0=0.05, 
            alpha_1=0.8, 
            size_0=0.1, 
            size_1=1.5
        )
        
        metrics = {
            'f1_score': f1,
            'train_distance': train_dis,
            'val_distance': val_dis,
            'test_distance': test_dis,
            'threshold': threshold
        }
        
        return metrics
    
    def _save_checkpoint(self, epoch: int, loss: float) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'memory_state_dict': self.memory.state_dict(),
            'gnn_state_dict': self.gnn.state_dict(),
            'deep_svdd_state_dict': self.deep_svdd.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'config': self.config
        }
        checkpoint_path = f'{self.results_path}/checkpoint_epoch{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f'Checkpoint saved: {checkpoint_path}')
    
    def run_training(self, day: str) -> Dict[str, Any]:
        """
        Run the complete training pipeline for a day.
        
        Args:
            day: Day identifier
            
        Returns:
            Dictionary containing training results and metrics
        """
        # Setup experiment
        self.setup_experiment(day)
        
        results = {
            'day': day,
            'epochs': [],
            'losses': [],
            'metrics': {}
        }
        
        # Training loop
        for epoch in range(1, self.config.n_epochs + 1):
            loss, c_param = self.train_epoch(epoch)
            self.logger.info(f'Epoch: {epoch:02d}, Loss: {loss:.6f}')
            
            results['epochs'].append(epoch)
            results['losses'].append(loss)
            
            # Evaluate at specified epochs
            if epoch in self.config.evaluation_epochs:
                metrics = self.evaluate_and_plot(epoch)
                results['metrics'][epoch] = metrics
                
                self.logger.info(f'Epoch {epoch} - F1: {metrics["f1_score"]:.4f}, '
                               f'Threshold: {metrics["threshold"]:.4f}')
        
        self.logger.info(f'Training completed for {day}')
        return results


def run_single_day_experiment(config: TGNSVDDConfig, day: str) -> Dict[str, Any]:
    """
    Run experiment for a single day.
    
    Args:
        config: Configuration object
        day: Day identifier
        
    Returns:
        Experiment results
    """
    experiment = TGNSVDDExperiment(config)
    return experiment.run_training(day)


def run_multi_day_experiments(config: TGNSVDDConfig) -> Dict[str, Dict[str, Any]]:
    """
    Run experiments for multiple days.
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary mapping day to experiment results
    """
    all_results = {}
    
    for day in config.days_to_run:
        print(f"\n{'='*60}")
        print(f"Starting experiment for: {day}")
        print(f"{'='*60}")
        
        try:
            results = run_single_day_experiment(config, day)
            all_results[day] = results
            print(f"✅ Completed experiment for: {day}")
            
        except Exception as e:
            print(f"❌ Failed experiment for {day}: {str(e)}")
            all_results[day] = {'error': str(e)}
    
    return all_results
