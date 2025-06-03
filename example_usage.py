"""
Example Usage of WINA-DGM Integration
"""

import torch
import torch.nn as nn
import numpy as np
from darwin_godel_machine import Agent, DarwinGodelMachine
from main import WINASelfOptimizer

def create_simple_model():
    """Create a simple neural network for demonstration"""
    return nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )

class TaskEnvironment:
    """Simple environment to evaluate WINA agents"""
    def __init__(self):
        self.model = create_simple_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Generate synthetic data
        self.X = torch.randn(1000, 784)  # 1000 samples of 784 features
        self.y = torch.randint(0, 10, (1000,))  # 10 classes
        
    def evaluate_agent(self, wina_agent):
        """Evaluate a WINA agent on the task"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        try:
            # Apply WINA masking to each linear layer
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    # Get weights and apply WINA masking
                    weights = module.weight.data.clone()
                    
                    # Ensure input dimension matches weight matrix
                    input_size = weights.size(1)
                    x = torch.randn(1, input_size, device=weights.device)  # Dummy input for masking
                    
                    # Apply WINA masking
                    mask, _ = wina_agent.compute_wina_mask(
                        x, 
                        weights,
                        wina_agent.sparsity_config['global']
                    )
                    
                    # Ensure mask has same device as weights
                    mask = mask.to(weights.device)
                    
                    # Apply mask to weights
                    module.weight.data = nn.Parameter(weights * mask)
            
            # Forward pass
            outputs = self.model(self.X)
            loss = self.criterion(outputs, self.y)
            
            # Metrics
            _, predicted = torch.max(outputs.data, 1)
            total = self.y.size(0)
            correct = (predicted == self.y).sum().item()
            accuracy = correct / total
            
            # Calculate FLOPs reduction
            original_flops = 784 * 256 + 256 * 128 + 128 * 10
            effective_flops = original_flops * (1 - wina_agent.sparsity_config['global'])
            flops_reduction = 1 - (effective_flops / original_flops)
            
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
    # Initialize environment
    env = TaskEnvironment()
    
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
    
    # Run evolution
    print("Starting WINA-DGM evolution...")
    dgm.evolve_population(
        n_generations=10,
        evaluation_task=env.evaluate_agent
    )
    
    # Get best agent
    best_agent, metrics = dgm.get_best_agent()
    print("\nEvolution complete!")
    print(f"Best accuracy: {metrics['accuracy']:.4f}")
    print(f"FLOPs reduction: {metrics['flops_reduction']*100:.2f}%")
    print("Final sparsity config:", best_agent.config)

if __name__ == "__main__":
    run_demo()
