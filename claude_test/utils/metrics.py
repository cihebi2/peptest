"""Evaluation Metrics"""

import torch
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score


def compute_metrics(predictions, targets, task_type='regression'):
    """
    Compute evaluation metrics
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        task_type: 'regression' or 'classification'
    
    Returns:
        metrics: Dictionary of metrics
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    
    metrics = {}
    
    if task_type == 'regression':
        # Pearson correlation
        if len(predictions.shape) > 1:
            predictions = predictions.squeeze()
        if len(targets.shape) > 1:
            targets = targets.squeeze()
        
        metrics['pearson_r'] = pearsonr(predictions, targets)[0]
        metrics['rmse'] = np.sqrt(mean_squared_error(targets, predictions))
        metrics['mae'] = np.mean(np.abs(predictions - targets))
    
    elif task_type == 'classification':
        # For binary classification
        if predictions.shape[-1] == 2:
            # Softmax probabilities
            probs = predictions[:, 1]
            preds = (probs > 0.5).astype(int)
        else:
            probs = predictions
            preds = (probs > 0.5).astype(int)
        
        metrics['accuracy'] = accuracy_score(targets, preds)
        metrics['auc'] = roc_auc_score(targets, probs)
    
    return metrics
