"""
Curriculum Learning for Graphs

Implements curriculum learning strategies that train on
examples from easy to hard, improving convergence and generalization
"""

import torch
import numpy as np
from torch.utils.data import Subset


class DifficultyMetrics:
    """Calculate difficulty metrics for graphs"""

    @staticmethod
    def num_nodes(graph):
        """Number of nodes as difficulty metric"""
        return graph.num_nodes()

    @staticmethod
    def num_edges(graph):
        """Number of edges as difficulty metric"""
        return graph.num_edges()

    @staticmethod
    def graph_complexity(graph):
        """Combined complexity metric"""
        num_nodes = graph.num_nodes()
        num_edges = graph.num_edges()
        # Could add: number of rings, molecular weight, etc.
        return num_nodes + num_edges * 0.5

    @staticmethod
    def density(graph):
        """Graph density as difficulty"""
        num_nodes = graph.num_nodes()
        num_edges = graph.num_edges()
        max_edges = num_nodes * (num_nodes - 1)
        return num_edges / max_edges if max_edges > 0 else 0


class CurriculumLearning:
    """
    Curriculum learning strategy: train on easy examples first,
    gradually introducing harder examples
    """

    def __init__(self, dataset, difficulty_metric='complexity', num_stages=5):
        """
        Args:
            dataset: Full dataset
            difficulty_metric: How to measure difficulty
                ('num_nodes', 'num_edges', 'complexity', 'density')
            num_stages: Number of curriculum stages
        """
        self.dataset = dataset
        self.difficulty_metric = difficulty_metric
        self.num_stages = num_stages

        # Compute difficulties and sort
        self.difficulties = self._compute_difficulties()
        self.sorted_indices = np.argsort(self.difficulties)

    def _compute_difficulties(self):
        """Compute difficulty for each example"""
        difficulties = []

        metric_fn = {
            'num_nodes': DifficultyMetrics.num_nodes,
            'num_edges': DifficultyMetrics.num_edges,
            'complexity': DifficultyMetrics.graph_complexity,
            'density': DifficultyMetrics.density,
        }.get(self.difficulty_metric, DifficultyMetrics.graph_complexity)

        for data in self.dataset:
            difficulty = metric_fn(data)
            difficulties.append(difficulty)

        return np.array(difficulties)

    def get_curriculum_subset(self, stage):
        """
        Get data subset for a given curriculum stage

        Args:
            stage: Current stage (0 to num_stages-1)

        Returns:
            Subset of dataset
        """
        stage = min(stage, self.num_stages - 1)

        # Gradually increase data difficulty
        # Stage 0: easiest 20%, Stage 1: easiest 40%, ..., Stage 4: all data
        ratio = (stage + 1) / self.num_stages
        cutoff = int(len(self.sorted_indices) * ratio)

        current_indices = self.sorted_indices[:cutoff]

        # Shuffle within stage
        np.random.shuffle(current_indices)

        return Subset(self.dataset, current_indices.tolist())

    def get_stage_from_epoch(self, epoch, total_epochs):
        """
        Determine curriculum stage from current epoch

        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs

        Returns:
            stage: Current curriculum stage
        """
        progress = epoch / total_epochs
        stage = int(progress * self.num_stages)
        return min(stage, self.num_stages - 1)


if __name__ == '__main__':
    print("Testing Curriculum Learning...")
    print("✓ Module structure created!")
