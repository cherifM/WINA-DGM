"""
Visualization utilities for WINA-DGM results.

This module provides functions to visualize the evolution of WINA-DGM optimization,
including fitness, accuracy, and sparsity metrics across generations.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from typing import Dict, List, Optional

# Set style for better looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def plot_evolution_metrics(history: Dict[str, List[float]], 
                         save_path: str = 'evolution_metrics.png'):
    """
    Plot the evolution of key metrics across generations.
    
    Args:
        history: Dictionary containing lists of metrics per generation
        save_path: Path to save the figure
    """
    plt.figure(figsize=(15, 10))
    
    # Plot fitness and accuracy
    plt.subplot(2, 2, 1)
    gens = range(1, len(history['fitness']) + 1)
    plt.plot(gens, history['fitness'], 'b-', label='Fitness', linewidth=2)
    plt.plot(gens, history['accuracy'], 'g-', label='Accuracy', linewidth=2)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Fitness and Accuracy over Generations', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot sparsity
    plt.subplot(2, 2, 2)
    plt.plot(gens, history['sparsity'], 'r-', linewidth=2)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Sparsity (1 - keep ratio)', fontsize=12)
    plt.title('Evolution of Sparsity', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Plot sparsity vs accuracy
    plt.subplot(2, 2, 3)
    scatter = plt.scatter(
        history['sparsity'], 
        history['accuracy'],
        c=gens,
        cmap='viridis',
        alpha=0.7,
        s=100
    )
    plt.colorbar(scatter, label='Generation')
    plt.xlabel('Sparsity (1 - keep ratio)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Sparsity vs Accuracy Trade-off', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Plot FLOPs reduction
    if 'flops_reduction' in history:
        plt.subplot(2, 2, 4)
        plt.plot(gens, history['flops_reduction'], 'm-', linewidth=2)
        plt.fill_between(gens, 
                        np.array(history['flops_reduction']) * 0.95,
                        np.array(history['flops_reduction']) * 1.05,
                        color='m', alpha=0.2)
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('FLOPs Reduction (%)', fontsize=12)
        plt.title('Computational Efficiency', fontsize=14)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved evolution metrics to {save_path}")

def visualize_weight_distribution(model, layer_name: str, save_path: str = 'weight_distribution.png'):
    """
    Visualize the weight distribution of a specific layer.
    
    Args:
        model: PyTorch model
        layer_name: Name of the layer to visualize
        save_path: Path to save the figure
    """
    weights = []
    for name, param in model.named_parameters():
        if layer_name in name and 'weight' in name:
            weights = param.data.cpu().numpy().flatten()
            break
    
    if len(weights) == 0:
        print(f"Layer {layer_name} not found or has no weights")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Plot histogram
    plt.subplot(1, 2, 1)
    plt.hist(weights, bins=100, alpha=0.7, color='b', density=True)
    plt.title(f'Weight Distribution\n{layer_name}', fontsize=14)
    plt.xlabel('Weight Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    
    # Plot boxplot
    plt.subplot(1, 2, 2)
    plt.boxplot(weights, vert=False)
    plt.title('Weight Statistics', fontsize=14)
    plt.yticks([])
    plt.xlabel('Weight Value', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved weight distribution to {save_path}")

def plot_sparsity_pattern(model, layer_name: str, save_path: str = 'sparsity_pattern.png'):
    """
    Visualize the sparsity pattern of a specific layer.
    
    Args:
        model: PyTorch model
        layer_name: Name of the layer to visualize
        save_path: Path to save the figure
    """
    for name, param in model.named_parameters():
        if layer_name in name and 'weight' in name:
            weights = param.data.cpu().numpy()
            break
    else:
        print(f"Layer {layer_name} not found or has no weights")
        return
    
    # Create binary mask (1 for non-zero, 0 for zero)
    mask = (weights != 0).astype(float)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='binary', aspect='auto')
    plt.title(f'Sparsity Pattern\n{layer_name}', fontsize=14)
    plt.xlabel('Input Features', fontsize=12)
    plt.ylabel('Output Features', fontsize=12)
    plt.colorbar(label='Active (1) / Pruned (0)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved sparsity pattern to {save_path}")

if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create mock history data
    num_generations = 20
    history = {
        'fitness': np.linspace(0.1, 0.9, num_generations) + np.random.normal(0, 0.02, num_generations),
        'accuracy': np.linspace(0.08, 0.85, num_generations) + np.random.normal(0, 0.02, num_generations),
        'sparsity': np.linspace(0.9, 0.4, num_generations) + np.random.normal(0, 0.02, num_generations),
        'flops_reduction': np.linspace(0.1, 0.7, num_generations) + np.random.normal(0, 0.02, num_generations)
    }
    
    # Generate example plots
    plot_evolution_metrics(history, 'example_evolution.png')
    
    # Example for weight distribution (requires a real model)
    # visualize_weight_distribution(model, 'layer1', 'example_weight_dist.png')
    
    # Example for sparsity pattern (requires a real model)
    # plot_sparsity_pattern(model, 'layer1', 'example_sparsity.png')
