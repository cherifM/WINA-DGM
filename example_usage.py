"""
Example Usage of WINA-DGM Integration
"""

import os
import torch
import torch.nn as nn
import numpy as np
import random
import gc
import copy
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from darwin_godel_machine import DarwinGodelMachine, Agent
from main import WINASelfOptimizer
from generate_report import WINAReportGenerator


def plot_evolution_metrics(fitness_history: List[float], 
                         accuracy_history: List[float],
                         sparsity_history: List[float],
                         avg_fitness_history: List[float],
                         save_path: Optional[str] = None):
    """Plot evolution metrics across generations.
    
    Args:
        fitness_history: List of best fitness values per generation
        accuracy_history: List of accuracy values per generation
        sparsity_history: List of sparsity values per generation
        avg_fitness_history: List of average fitness values per generation
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(15, 10))
    
    # Plot fitness and accuracy
    plt.subplot(2, 2, 1)
    gens = range(1, len(fitness_history) + 1)
    plt.plot(gens, fitness_history, 'b-', label='Best Fitness', linewidth=2)
    plt.plot(gens, avg_fitness_history, 'g--', label='Avg Fitness', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(gens, accuracy_history, 'r-', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Evolution')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    
    # Plot sparsity
    plt.subplot(2, 2, 3)
    plt.plot(gens, sparsity_history, 'm-', linewidth=2)
    plt.xlabel('Generation')
    plt.ylabel('Sparsity')
    plt.title('Sparsity Evolution')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    
    # Plot fitness vs sparsity
    plt.subplot(2, 2, 4)
    plt.scatter(sparsity_history, fitness_history, c=gens, cmap='viridis')
    plt.colorbar(label='Generation')
    plt.xlabel('Sparsity')
    plt.ylabel('Fitness')
    plt.title('Fitness vs Sparsity')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved evolution plot to {save_path}")
    
    plt.close()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    
# Function to clean up GPU memory
def clean_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def create_enhanced_model():
    """Create an enhanced neural network for demonstration"""
    model = nn.Sequential(
        nn.Linear(784, 512),
        nn.BatchNorm1d(512),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.3),
        
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.3),
        
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.LeakyReLU(0.1),
        nn.Dropout(0.2),
        
        nn.Linear(128, 10)
    )
    
    # Initialize weights
    for layer in model:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
    return model

class TaskEnvironment:
    """Simple environment to evaluate WINA agents"""
    def __init__(self, input_size=784, hidden_size=512, output_size=10, device='cuda'):
        self.model = create_enhanced_model().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.device = device
        
        # Create random data for demonstration
        self.X = torch.randn(1000, input_size, device=device)
        self.y = torch.randint(0, output_size, (1000,), device=device)  # 10 classes
        
    def evaluate_agent(self, wina_agent):
        """Evaluate a WINA agent on the task"""
        self.model.train()
        
        try:
            # Apply WINA masking to each linear layer
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    # Get weights and ensure they're on the right device
                    weights = module.weight.data.clone()
                    
                    # Create dummy input matching the layer's input dimensions
                    input_size = weights.size(1)
                    batch_size = 32  # Small batch for evaluation
                    x = torch.randn(batch_size, input_size, device=weights.device)
                    
                    # Apply WINA masking
                    mask, _ = wina_agent.compute_wina_mask(
                        x,  
                        weights,
                        wina_agent.sparsity_config['global']
                    )
                    
                    # Ensure mask has the right shape (1, input_features) -> (output_features, input_features)
                    if mask.dim() == 2 and mask.size(0) == 1:
                        mask = mask.expand(weights.size(0), -1)
                    
                    # Apply mask to weights
                    module.weight.data = weights * mask.to(weights.device)
            
            # Forward pass with gradient tracking disabled for evaluation
            with torch.no_grad():
                outputs = self.model(self.X)
                loss = self.criterion(outputs, self.y)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total = self.y.size(0)
                correct = (predicted == self.y).sum().item()
                accuracy = correct / total
                
                # Calculate FLOPs reduction
                original_flops = 784 * 256 + 256 * 128 + 128 * 10
                effective_flops = original_flops * (1 - wina_agent.sparsity_config['global'])
                flops_reduction = 1 - (effective_flops / original_flops)
                
                # Clean up
                del outputs, predicted
                clean_memory()
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            # Return minimum metrics in case of error
            accuracy = 0.0
            loss = float('inf')
            flops_reduction = 0.0
        
        return {
            'accuracy': accuracy,
            'loss': loss.item(),
            'flops_reduction': flops_reduction,
            'fitness': accuracy + 0.5 * flops_reduction  # Combined metric
        }

def run_demo():
    """Run the WINA-DGM demo"""
    # Clean up any existing memory
    clean_memory()
    
    # Initialize environment
    env = TaskEnvironment()
    
    # Print model summary
    print("\nModel Architecture:")
    print(env.model)
    
    # Print device information
    print(f"\nModel is on device: {next(env.model.parameters()).device}")
    print(f"Input data is on device: {env.X.device}")
    
    # Create enhanced model with better architecture
    model = create_enhanced_model().to(device)
    
    # Create initial configuration with layer-wise sparsity targets
    initial_config = {
        'global': 0.5,  # 50% sparsity target
        'fc1': 0.3,    # 30% sparsity for first layer
        'fc2': 0.5,    # 50% sparsity for second layer
        'fc3': 0.7,    # 70% sparsity for third layer
        'importance_threshold': 0.01,
        'learning_rate': 0.001
    }
    
    # Create initial agent
    initial_agent = Agent(
        id="agent_0",
        model=model,  # Pass the model to the agent
        config=initial_config
    )
    
    # Initialize DGM with enhanced parameters
    dgm = DarwinGodelMachine(
        initial_agent=initial_agent,
        population_size=20,         # Larger population for better exploration
        novelty_weight=0.4,         # Higher weight for diversity
        mutation_rate=0.15,         # Slightly higher mutation rate
        elite_size=3,               # Keep more elites
        crossover_prob=0.7,         # Probability of crossover
        min_sparsity=0.1,           # Minimum allowed sparsity
        max_sparsity=0.95,          # Maximum allowed sparsity
        device=device               # Use the same device as the model
    )
    
    # Track evolution metrics
    best_fitness_history = []
    best_accuracy_history = []
    best_sparsity_history = []
    
    # Initialize report generator
    report_gen = WINAReportGenerator(output_dir='wina_reports')
    
    # Track metrics across generations
    all_metrics = {
        'fitness': [],
        'accuracy': [],
        'sparsity': [],
        'flops_reduction': []
    }
    
    # Run evolution with progress tracking
    print("Starting WINA-DGM evolution...")
    # Initialize population if not already done
    if not hasattr(dgm, 'population') or not dgm.population:
        dgm._initialize_population()
    
    # Track evolution metrics
    fitness_history = []
    accuracy_history = []
    sparsity_history = []
    avg_fitness_history = []
    
    # Evolution loop
    print("Starting WINA-DGM evolution...\n")
    
    def evaluate_agent(agent):
        try:
            # Get the target sparsity from agent config
            target_sparsity = agent.config.get('global', 0.7)
            
            # Apply sparsity to model weights using compute_wina_mask
            with torch.no_grad():
                # Create a forward pass to get intermediate activations
                model = agent.wina.model
                
                # Store original weights
                original_weights = {}
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        original_weights[name] = param.data.clone()
                
                # Apply sparsity to each linear layer
                for name, module in model.named_modules():
                    if isinstance(module, nn.Linear):
                        # Get weights
                        weights = module.weight.data
                        
                        # Create random input with correct dimensions for this layer
                        input_size = weights.size(1)  # input features
                        x = torch.randn(1, input_size, device=weights.device)
                        
                        try:
                            # Compute importance mask
                            mask, _ = agent.wina.compute_wina_mask(
                                x,
                                weights,
                                target_sparsity
                            )
                            
                            # Apply mask to weights
                            module.weight.data = weights * mask.to(weights.device)
                            
                        except Exception as e:
                            print(f"Error in layer {name}: {e}")
                            continue
                
                # Now evaluate the model with sparsity applied
                try:
                    outputs = model(env.X)
                    _, predicted = torch.max(outputs.data, 1)
                    correct = (predicted == env.y).sum().item()
                    accuracy = correct / len(env.y)
                    
                    # Calculate actual sparsity (fraction of weights that are zero)
                    total_weights = 0
                    zero_weights = 0
                    for p in model.parameters():
                        if p.dim() > 1:  # Only consider weight matrices
                            total_weights += p.numel()
                            zero_weights += (p == 0).sum().item()
                    
                    actual_sparsity = zero_weights / total_weights if total_weights > 0 else 0.0
                    
                    # Fitness is weighted combination of accuracy and sparsity
                    fitness = 0.7 * accuracy + 0.3 * actual_sparsity
                    
                    return fitness, accuracy, actual_sparsity
                    
                except Exception as e:
                    print(f"Error during model evaluation: {e}")
                    # Restore original weights
                    for name, param in model.named_parameters():
                        if 'weight' in name and name in original_weights:
                            param.data.copy_(original_weights[name])
                    return 0.0, 0.0, 0.0
                
        except Exception as e:
            print(f"Error in evaluate_agent: {e}")
            return 0.0, 0.0, 0.0
    
    # Evolution loop
    for gen in range(10):  # 10 generations
        print(f"\n--- Generation {gen+1}/10 ---")
        
        # Track metrics for this generation
        gen_fitness = []
        best_fitness = 0
        best_accuracy = 0
        best_sparsity = 0
        
        # Evaluate all agents in current population
        for agent in dgm.population:
            fitness, accuracy, sparsity = evaluate_agent(agent)
            agent.fitness = fitness
            gen_fitness.append(fitness)
            
            # Track best agent in this generation
            if fitness > best_fitness:
                best_fitness = fitness
                best_accuracy = accuracy
                best_sparsity = sparsity
            
            print(f"Agent {agent.id}: Fitness={fitness:.4f}, "
                  f"Accuracy={accuracy*100:.2f}%, "
                  f"Sparsity={sparsity*100:.1f}%")
        
        # Track metrics across generations
        fitness_history.append(best_fitness)
        accuracy_history.append(best_accuracy)
        sparsity_history.append(best_sparsity)
        
        # Calculate average fitness for this generation
        avg_fitness = sum(gen_fitness) / len(gen_fitness)
        avg_fitness_history.append(avg_fitness)
        
        # Print generation summary
        print(f"\nGeneration {gen+1} Summary:")
        print(f"Best fitness: {best_fitness:.4f}")
        print(f"Best accuracy: {best_accuracy*100:.2f}%")
        print(f"Best sparsity: {best_sparsity*100:.1f}%")
        print(f"Average fitness: {avg_fitness:.4f}")
        
        # Generate intermediate visualizations every few generations
        if (gen + 1) % 2 == 0 or gen == 9:  # More frequent updates
            plot_evolution_metrics(
                fitness_history, 
                accuracy_history, 
                sparsity_history,
                avg_fitness_history,
                save_path=f"wina_reports/evolution_gen_{gen+1}.png"
            )
        
        # Create next generation (if not the last generation)
        if gen < 9:
            dgm._select_and_reproduce()
    
    # Ensure we have the final best agent
    best_agent = max(dgm.population, key=lambda x: x.fitness)
    
    print("\nEvolution complete!")
    print("\n=== Final Results ===")
    print(f"Best fitness: {best_agent.fitness:.4f}")
    print(f"Best accuracy: {(best_agent.fitness - 0.3 * best_agent.config.get('global', 0.7)) / 0.7 * 100:.2f}%")
    print(f"Best sparsity: {1 - best_agent.config.get('global', 0.7):.1%}")
    
    # Save the best agent
    model_path = os.path.join('wina_reports', 'best_wina_agent.pth')
    if hasattr(best_agent.wina, 'state_dict'):
        torch.save({
            'agent_state_dict': best_agent.wina.state_dict(),
            'sparsity_config': best_agent.config,
            'metrics': {
                'fitness': best_agent.fitness,
                'accuracy': (best_agent.fitness - 0.3 * best_agent.config.get('global', 0.7)) / 0.7,
                'sparsity': 1 - best_agent.config.get('global', 0.7)
            }
        }, model_path)
        print(f"\nSaved best agent to '{model_path}'")
    
    # Generate final visualizations
    print("\nGenerating final visualizations...")
    
    # 1. Evolution of metrics
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    
    gens = range(1, len(fitness_history) + 1)
    ax1.plot(gens, fitness_history, 'b-', label='Fitness')
    ax1.plot(gens, accuracy_history, 'g-', label='Accuracy')
    ax2.plot(gens, sparsity_history, 'r--', label='Sparsity')
    
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Score', color='b')
    ax2.set_ylabel('Sparsity', color='r')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title('Final Evolution Metrics')
    
    report_gen.add_figure(fig1, 'Final evolution of fitness, accuracy, and sparsity', 'final_evolution')
    plt.close(fig1)
    
    # 2. Sparsity vs Accuracy trade-off
    fig2, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        sparsity_history, 
        accuracy_history,
        c=gens,
        cmap='viridis',
        alpha=0.7,
        s=100
    )
    plt.colorbar(scatter, label='Generation')
    ax.set_xlabel('Sparsity (1 - keep ratio)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Sparsity vs Accuracy Trade-off')
    ax.grid(True, alpha=0.3)
    
    report_gen.add_figure(fig2, 'Sparsity vs Accuracy trade-off across generations', 'sparsity_vs_accuracy')
    plt.close(fig2)
    
    # 3. FLOPs reduction vs Accuracy
    fig3, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        [1 - s for s in sparsity_history],
        accuracy_history,
        c=gens,
        cmap='plasma',
        alpha=0.7,
        s=100
    )
    plt.colorbar(scatter, label='Generation')
    ax.set_xlabel('FLOPs Reduction')
    ax.set_ylabel('Accuracy')
    ax.set_title('Computational Efficiency vs Accuracy')
    ax.grid(True, alpha=0.3)
    
    report_gen.add_figure(fig3, 'Computational efficiency vs Accuracy trade-off', 'flops_vs_accuracy')
    plt.close(fig3)
    
    # Generate and save the final report
    print("\nGenerating final report...")
    report_path = report_gen.generate_report(all_metrics)
    
    print("\n" + "="*80)
    print(f"WINA-DGM Optimization Complete!")
    print("="*80)
    print(f"Best fitness: {best_agent.fitness:.4f}")
    print(f"Best accuracy: {best_agent.performance:.2%}")
    print(f"Best sparsity: {best_agent.config.get('global', 0.5):.1%}")
    print(f"\nReport saved to: {os.path.abspath(report_path)}")
    print(f"Model saved to: {os.path.abspath(model_path)}")
    print("="*80)

if __name__ == "__main__":
    run_demo()
