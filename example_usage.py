"""
Example Usage of WINA-DGM Integration
"""

import torch
import torch.nn as nn
import numpy as np
import gc
from darwin_godel_machine import Agent, DarwinGodelMachine
from main import WINASelfOptimizer

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

def create_simple_model():
    """Create a simple neural network for demonstration"""
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.2),  # Add dropout for better generalization
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.2),  # Add dropout for better generalization
        nn.Linear(128, 10)
    )
    
    # Initialize weights using Kaiming initialization
    for layer in model:
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
    
    return model

class TaskEnvironment:
    """Simple environment to evaluate WINA agents"""
    def __init__(self):
        self.model = create_simple_model().to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Generate synthetic data
        self.X = torch.randn(1000, 784, device=device)  # 1000 samples of 784 features
        self.y = torch.randint(0, 10, (1000,), device=device)  # 10 classes
        
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
    
    # Create initial agent
    initial_config = {
        'global': 0.5,
        'layer_schedule': [0.4, 0.5, 0.6],
        'importance_threshold': 0.1
    }
    initial_agent = Agent(
        id="initial_agent",
        config=initial_config
    )
    
    # Initialize DGM
    dgm = DarwinGodelMachine(
        initial_agent=initial_agent,
        population_size=20,
        novelty_weight=0.3,
        mutation_rate=0.15,
        elite_size=2
    )
    
    # Track evolution metrics
    best_fitness_history = []
    best_accuracy_history = []
    best_sparsity_history = []
    
    # Run evolution with progress tracking
    print("Starting WINA-DGM evolution...")
    
    # Initialize population if not already done
    if not hasattr(dgm, 'population') or not dgm.population:
        dgm._initialize_population()
    
    # Run evolution loop
    for gen in range(10):  # 10 generations
        print(f"\n--- Generation {gen + 1}/10 ---")
        
        # Evaluate all agents in current population
        for agent in dgm.population:
            try:
                metrics = env.evaluate_agent(agent.wina)
                agent.performance = metrics.get('accuracy', 0)
                agent.fitness = agent.performance + 0.3 * metrics.get('flops_reduction', 0)
            except Exception as e:
                print(f"Error evaluating agent: {e}")
                agent.performance = 0
                agent.fitness = 0
        
        # Track best agent in this generation
        gen_best = max(dgm.population, key=lambda x: x.fitness)
        best_fitness_history.append(gen_best.fitness)
        best_accuracy_history.append(gen_best.performance)
        best_sparsity_history.append(gen_best.config.get('global', 0.5))
        
        print(f"Best fitness: {gen_best.fitness:.4f}")
        print(f"Best accuracy: {gen_best.performance:.2%}")
        print(f"Sparsity: {gen_best.config.get('global', 0.5):.1%}")
        
        # Create next generation (simplified - in a real scenario, use DGM's evolution logic)
        if gen < 9:  # Don't evolve after the last generation
            # Sort by fitness and keep top 50%
            dgm.population.sort(key=lambda x: x.fitness, reverse=True)
            dgm.population = dgm.population[:len(dgm.population)//2]
            
            # Repopulate with mutations of the best agents
            while len(dgm.population) < 20:  # Assuming population size of 20
                parent = random.choice(dgm.population)
                child = dgm._mutate_agent(parent)
                dgm.population.append(child)
    
    # Get overall best agent
    best_agent = max(dgm.population, key=lambda x: x.fitness)
    
    print("\nEvolution complete!")
    print("\n=== Final Results ===")
    print(f"Best fitness: {best_agent.fitness:.4f}")
    print(f"Best accuracy: {best_agent.performance:.2%}")
    print(f"Best sparsity: {best_agent.config.get('global', 0.5):.1%}")
    
    # Save the best agent
    if hasattr(best_agent.wina, 'state_dict'):
        torch.save({
            'agent_state_dict': best_agent.wina.state_dict(),
            'sparsity_config': best_agent.config,
            'metrics': {
                'fitness': best_agent.fitness,
                'accuracy': best_agent.performance,
                'sparsity': best_agent.config.get('global', 0.5)
            }
        }, 'best_wina_agent.pth')
        print("\nSaved best agent to 'best_wina_agent.pth'")
    
    # Plot evolution history
    import matplotlib.pyplot as plt
    
    # Create figure with two subplots
    plt.figure(figsize=(15, 5))
    
    # Plot fitness and accuracy
    plt.subplot(1, 2, 1)
    gens = range(1, len(best_fitness_history) + 1)
    plt.plot(gens, best_fitness_history, 'b-', label='Fitness')
    plt.plot(gens, best_accuracy_history, 'g-', label='Accuracy')
    plt.xlabel('Generation')
    plt.ylabel('Value')
    plt.title('Evolution of Fitness and Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot sparsity
    plt.subplot(1, 2, 2)
    plt.plot(gens, best_sparsity_history, 'r-')
    plt.xlabel('Generation')
    plt.ylabel('Sparsity (1 - keep ratio)')
    plt.title('Evolution of Sparsity')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('evolution_metrics.png')
    print("\nSaved evolution metrics to 'evolution_metrics.png'")
    
    # Create a second figure for sparsity vs accuracy
    plt.figure(figsize=(10, 5))
    plt.scatter(best_sparsity_history, best_accuracy_history, c='r', alpha=0.5)
    plt.title('Sparsity vs Accuracy')
    plt.xlabel('Sparsity (1 - keep ratio)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('sparsity_vs_accuracy.png')
    print("Saved sparsity vs accuracy plot to 'sparsity_vs_accuracy.png'")
        plt.title('Sparsity vs Accuracy by Generation')
        plt.grid(True, alpha=0.3)
        plt.savefig('sparsity_vs_accuracy.png')
        
        print("\nSaved evolution metrics to 'evolution_metrics.png' and 'sparsity_vs_accuracy.png'")

if __name__ == "__main__":
    run_demo()
