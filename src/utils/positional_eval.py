#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEPRECATED: Optional utility not required for the minimal experiment path.
Canonical entry: `src/main.py` → `experiments.tgn_svdd_experiment`.
Kept for reference only.
"""

import pandas as pd
import torch
import numpy as np



def get_position_regr_3(batch_nodes, all_nodes, score_func, sort_reverse=True):
    src_indices = batch_nodes[:, 0].long()
    dst_indices = batch_nodes[:, 1].long()
    src_embeddings = batch_nodes[:, 2:]

    # scores = score_func(src_embeddings, all_nodes[:, 1:])
    scores = score_func(src_embeddings.unsqueeze(1), all_nodes[: , 1:].unsqueeze(0))
    
    scores[list(range(len(src_indices))), src_indices] = float('-inf') if sort_reverse else float('inf')
    sorted_scores, sorted_indices = torch.sort(scores, dim=1, descending=sort_reverse)

    positions = (sorted_indices == dst_indices[:, None]).nonzero()[:, 1]
    return positions

def get_position_clf(batch_nodes, all_nodes, score_func, sort_reverse=True):
    src_indices = batch_nodes[:, 0].long()
    dst_indices = batch_nodes[:, 1].long()
    src_embeddings = batch_nodes[:, 2:]

    # scores = score_func(src_embeddings, all_nodes[:, 1:])
    scores = score_func(src_embeddings.unsqueeze(1), all_nodes[: , 1:].unsqueeze(0)).sigmoid()
    
    scores[list(range(len(src_indices))), src_indices] = float('-inf') if sort_reverse else float('inf')
    sorted_scores, sorted_indices = torch.sort(scores, dim=1, descending=sort_reverse)

    positions = (sorted_indices == dst_indices[:, None, None]).nonzero()[:, 1]
    return positions




def get_position_regr(batch_nodes, all_nodes, score_func, sort_reverse=True):
    # for cosine sort reverse, for mse sort normaly sort_reverse=False
    result = []
    for i in range(batch_nodes.shape[0]):
        row_batch = batch_nodes[i]
        
        dst_index = row_batch[1].item()
        src_index = row_batch[0].item()
        src_embedding = row_batch[2:]

        # scores = [score_func(src_embedding.unsqueeze(0), row[1:].unsqueeze(0)) for row in all_nodes]
        
        scores = score_func(src_embedding.unsqueeze(0), all_nodes[: , 1:])
        
        index_score_dict = dict(zip(all_nodes[:, 0].tolist(), scores))
        index_score_dict.pop(src_index, None)  # exclude the entry with src_index from the dictionary
        sorted_index_score = sorted(index_score_dict.items(), key=lambda x: x[1], reverse=sort_reverse)

        position = [i for i, (index, score) in enumerate(sorted_index_score) if index == dst_index][0]
        result.append(position)

    return np.array(result)



# tests of the ranking function
def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item()

def cosine_similarity_2(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=2)#.item()

def cosine_similarity_for(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=1)

