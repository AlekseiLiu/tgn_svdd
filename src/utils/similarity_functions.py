#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPRECATED: Optional similarity helpers not required for the minimal experiment.
Canonical entry: `src/main.py` → `experiments.tgn_svdd_experiment`.
Kept for reference and legacy script compatibility.
"""

# import pandas as pd
import torch
import numpy as np






def cosine_similarity(pred, target):
    dot = torch.sum(pred * target, dim=1)
    pred_norm = torch.norm(pred, p=2, dim=1)
    target_norm = torch.norm(target, p=2, dim=1)
    return dot / (pred_norm * target_norm)

# this one for ranking!!!
def cosine_similarity_2(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=2)#.item() choose dim for better bradcasting


# torch.nn.functional.cosine_similarity(a, b).item()

def pearson_r(pred, target):
    pred_mean = torch.mean(pred, dim=0)
    target_mean = torch.mean(target, dim=0)
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    covariance = torch.mean(pred_centered * target_centered, dim=0)
    pred_std = torch.std(pred, dim=0)
    target_std = torch.std(target, dim=0)
    return covariance / (pred_std * target_std)

def mse(pred, target):
    return  torch.mean((pred - target) ** 2, dim=1)
def mse_rank(pred, target):
    return  torch.mean((pred - target) ** 2, dim=2)

def mse_3(pred, target):
    return torch.nn.functional.mse_loss(pred, target)














