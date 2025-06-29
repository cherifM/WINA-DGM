"""
WINA-DGM Integration Example: Self-Optimizing Sparse Activation
==============================================================

This example demonstrates how a DGM agent can autonomously optimize
its WINA sparsity configuration through self-modification.
"""

import torch
import numpy as np
from typing import Dict, Tuple
import ast

class WINASelfOptimizer:
    """
    WINA (Weight Importance and Neuron Activation) Self-Optimizer
    
    Implements a self-optimizing neural network using weight importance and activation patterns
    to learn optimal sparsity masks through evolutionary optimization.
    
    Key Mathematical Formulations:
    - Weight Importance: I_ij = |w_ij| ⋅ 𝔼[|x_i|]
    - Sparsity Constraint: ∑(1 - m_ij) ≤ γ⋅n  where γ ∈ [0,1] is target sparsity
    - Orthogonality Loss: L_orth = β⋅||W^T W - I||_F^2
    - Fitness: F = accuracy + λ⋅FLOPs_reduction
    
    Args:
        model: PyTorch model to optimize
        sparsity_config: Dict of layer_name: target_sparsity
        device: Device to run optimization on (cuda/cpu)
    """
    def __init__(self, model, sparsity_config=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.device = device
        self.sparsity_config = sparsity_config or {'global': 0.5}  # γ: target sparsity ratio
        
        # Hyperparameters
        self.α = 0.01  # Learning rate for WINA updates
        self.β = 0.1   # Orthogonalization strength
        self.λ = 0.3   # FLOPs reduction weight in fitness
        
        # Tracking metrics
        self.metrics = {
            'sparsity': [],      # 1 - |m|₀/n where n is total params
            'orthogonality': [],  # ||W^T W - I||_F
            'fitness': [],       # Accuracy + λ⋅FLOPs_reduction
            'accuracy': [],      # Task accuracy
            'flops_red': []      # FLOPs reduction percentage
        }
    
    def compute_wina_mask(self, x: torch.Tensor, weights: torch.Tensor, sparsity: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute WINA mask based on weight and input importance.
        
        Args:
            x: Input activations of shape (batch_size, in_features)
            weights: Weight matrix of shape (out_features, in_features)
            sparsity: Target sparsity ratio γ ∈ [0,1]
            
        Returns:
            mask: Binary mask tensor (1, in_features)
            importance: Importance scores (1, in_features)
            
        Math:
            1. Compute weight importance: I_ij = |w_ij| ⋅ 𝔼[|x_i|]
            2. Keep top-k important weights: k = ⌈(1-γ)⋅n⌉
            3. Return binary mask m ∈ {0,1}^n
        """
        with torch.no_grad():
            try:
                # Reshape and move to correct device
                weights = weights.view(weights.size(0), -1).to(x.device)
                input_dim = weights.size(1)
                
                # 1. Compute importance scores
                weight_importance = torch.norm(weights, dim=0, keepdim=True)  # L2 norm over output dim
                input_importance = torch.abs(x).mean(dim=0, keepdim=True)     # Mean abs activation
                importance = weight_importance * input_importance  # Element-wise product
                
                # 2. Compute top-k threshold
                k = max(1, int((1 - sparsity) * input_dim))  # At least 1 connection
                
                # 3. Create binary mask
                if k < input_dim:
                    _, topk_indices = torch.topk(importance, k, dim=1)
                    mask = torch.zeros_like(importance, device=x.device)
                    mask.scatter_(1, topk_indices, 1.0)  # One-hot mask
                else:
                    mask = torch.ones_like(importance, device=x.device)
                    
                return mask, importance
                
            except Exception as e:
                print(f"Error in compute_wina_mask: {str(e)}")
                # Fallback: return all-ones mask
                input_dim = weights.size(1) if len(weights.shape) > 1 else weights.size(0)
                return (torch.ones(1, input_dim, device=x.device), 
                        torch.ones(1, input_dim, device=x.device))
    
    def analyze_sparsity_impact(self, task_data: Dict) -> Dict[str, float]:
        """Analyze impact of current sparsity configuration"""
        x = task_data['input']
        weights = task_data['weights']
        
        metrics = {}
        
        # Test different sparsity levels
        for sparsity in [0.5, 0.65, 0.8]:
            mask, importance = self.compute_wina_mask(x, weights, sparsity)
            
            # Measure impact
            active_ratio = mask.mean().item()
            importance_variance = importance.var().item()
            
            # Estimate error (simplified)
            pruned_importance = importance * (1 - mask)
            estimated_error = pruned_importance.sum().item()
            
            metrics[f'sparsity_{sparsity}'] = {
                'active_neurons': active_ratio,
                'importance_var': importance_variance,
                'estimated_error': estimated_error,
                'flops_reduction': 1 - active_ratio
            }
        
        return metrics
    
    def generate_sparsity_mutation(self) -> str:
        """Generate code to modify sparsity configuration"""
        # Analyze recent performance
        if len(self.performance_history) > 5:
            recent_perf = self.performance_history[-5:]
            trend = np.polyfit(range(5), recent_perf, 1)[0]
            
            if trend < 0:  # Performance decreasing
                # Reduce sparsity (activate more neurons)
                delta = -0.05
            else:  # Performance increasing
                # Try increasing sparsity
                delta = 0.05
        else:
            # Random exploration
            delta = np.random.uniform(-0.1, 0.1)
        
        new_sparsity = np.clip(self.sparsity_config['global'] + delta, 0.3, 0.9)
        
        # Generate mutation code
        mutation_code = f"""
# Self-modification: Adjust WINA sparsity
self.sparsity_config['global'] = {new_sparsity}

# Update layer schedule with gradient
schedule = self.sparsity_config['layer_schedule']
for i in range(len(schedule)):
    # Higher sparsity in middle layers
    if i < len(schedule) // 3 or i > 2 * len(schedule) // 3:
        schedule[i] = {new_sparsity - 0.1}
    else:
        schedule[i] = {new_sparsity + 0.05}

# Add performance-based adaptation
if hasattr(self, 'model'):
    for idx, layer in enumerate(self.model.layers):
        if idx < len(schedule):
            layer.set_sparsity(schedule[idx])
"""
        return mutation_code
    
    def optimize_orthogonalization_frequency(self) -> str:
        """Generate code to optimize when to recompute orthogonalization"""
        optimization_code = """
# Self-modification: Adaptive orthogonalization
class AdaptiveWINALayer(WINALayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ortho_counter = 0
        self.ortho_frequency = 10  # Recompute every N forward passes
        
    def forward(self, x):
        # Only recompute orthogonalization periodically
        if self.ortho_counter % self.ortho_frequency == 0:
            self.compute_orthogonalization()
        self.ortho_counter += 1
        
        # Rest of forward pass...
        return super().forward(x)

# Replace layers with adaptive version
for name, module in self.model.named_modules():
    if isinstance(module, WINALayer):
        adaptive = AdaptiveWINALayer(
            module.in_features, 
            module.out_features,
            module.sparsity
        )
        setattr(self.model, name, adaptive)
"""
        return optimization_code
    
    def theoretical_error_bound(self, sparsity: float, 
                               weight_matrix: torch.Tensor) -> float:
        """Compute theoretical error bound for WINA sparsity"""
        # Based on WINA paper Theorem 3.5
        n, m = weight_matrix.shape
        
        # Compute spectral norm
        spectral_norm = torch.linalg.norm(weight_matrix, ord=2).item()
        
        # Simplified error bound
        # E[||y_WINA - y||^2] ≤ (sparsity * spectral_norm^2) / sqrt(m)
        error_bound = (sparsity * spectral_norm**2) / np.sqrt(m)
        
        return error_bound
    
    def evolve_sparsity_strategy(self, performance_data: Dict) -> Tuple[str, Dict]:
        """Main evolution step: analyze and modify sparsity strategy"""
        
        # 1. Analyze current performance
        current_accuracy = performance_data.get('accuracy', 0)
        current_flops = performance_data.get('flops_reduction', 0)
        
        # 2. Compute fitness (balance accuracy and efficiency)
        fitness = current_accuracy - 0.1 * performance_data.get('inference_time', 1)
        fitness += 0.2 * current_flops  # Reward efficiency
        
        self.performance_history.append(fitness)
        
        # 3. Decide mutation strategy
        if current_flops < 0.5:  # Not sparse enough
            strategy = 'increase_sparsity'
        elif current_accuracy < 0.8:  # Accuracy suffering
            strategy = 'reduce_sparsity'
        else:  # Good balance, try layer-specific optimization
            strategy = 'optimize_layers'
        
        # 4. Generate appropriate mutation
        if strategy == 'increase_sparsity':
            mutation = self.generate_sparsity_mutation()
        elif strategy == 'reduce_sparsity':
            self.sparsity_config['global'] *= 0.9
            mutation = f"self.sparsity_config['global'] = {self.sparsity_config['global']}"
        else:
            mutation = self.optimize_orthogonalization_frequency()
        
        # 5. Theoretical validation
        test_weights = torch.randn(768, 768)  # Example weights
        error_bound = self.theoretical_error_bound(
            self.sparsity_config['global'], 
            test_weights
        )
        
        # 6. Return mutation and updated config
        return mutation, {
            'strategy': strategy,
            'new_sparsity': self.sparsity_config['global'],
            'theoretical_error': error_bound,
            'expected_flops_reduction': self.sparsity_config['global']
        }


# ===========================
# Demonstration
# ===========================

def demonstrate_wina_dgm_synergy():
    """Show how WINA and DGM work together"""
    
    print("=== WINA-DGM Self-Optimization Demo ===\n")
    
    # Create self-optimizing agent
    agent = WINASelfOptimizer()
    
    # Simulate evolution cycles
    for cycle in range(5):
        print(f"--- Evolution Cycle {cycle + 1} ---")
        
        # Generate test data
        test_data = {
            'input': torch.randn(32, 768),
            'weights': torch.randn(768, 3072)
        }
        
        # Analyze current configuration
        impact_metrics = agent.analyze_sparsity_impact(test_data)
        
        # Simulate performance (would be actual benchmark in practice)
        simulated_performance = {
            'accuracy': 0.75 + 0.05 * np.random.randn(),
            'inference_time': 1.0 - 0.5 * agent.sparsity_config['global'],
            'flops_reduction': agent.sparsity_config['global']
        }
        
        # Evolve sparsity strategy
        mutation_code, evolution_info = agent.evolve_sparsity_strategy(
            simulated_performance
        )
        
        print(f"Current Sparsity: {agent.sparsity_config['global']:.3f}")
        print(f"Strategy: {evolution_info['strategy']}")
        print(f"Theoretical Error Bound: {evolution_info['theoretical_error']:.4f}")
        print(f"Expected FLOPS Reduction: {evolution_info['expected_flops_reduction']:.1%}")
        
        # Show generated mutation
        print(f"\nGenerated Mutation:")
        print("```python")
        print(mutation_code.strip())
        print("```")
        
        # Execute mutation (in real DGM, this would modify the agent's code)
        exec(mutation_code, {'self': agent, 'WINALayer': None, 'np': np})
        
        print()
    
    # Final analysis
    print("=== Final Configuration ===")
    print(f"Global Sparsity: {agent.sparsity_config['global']:.3f}")
    print(f"Layer Schedule: {[f'{s:.2f}' for s in agent.sparsity_config['layer_schedule']]}")
    
    # Performance trajectory
    if agent.performance_history:
        print(f"\nPerformance Trajectory:")
        for i, perf in enumerate(agent.performance_history):
            print(f"  Cycle {i+1}: {perf:.4f}")
        
        improvement = agent.performance_history[-1] - agent.performance_history[0]
        print(f"\nTotal Improvement: {improvement:+.4f}")


# ===========================
# Advanced Integration Example
# ===========================

class WINACodeTransformer:
    """Transforms agent code to integrate WINA optimizations"""
    
    @staticmethod
    def inject_dynamic_sparsity(code: str) -> str:
        """Inject dynamic sparsity adaptation into agent code"""
        
        # Parse the code
        tree = ast.parse(code)
        
        # Find forward methods and inject WINA logic
        class WINAInjector(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                if node.name == 'forward':
                    # Add sparsity adaptation logic
                    adapt_code = ast.parse("""
# Dynamic sparsity based on input statistics
input_variance = x.var(dim=-1, keepdim=True)
adaptive_sparsity = self.base_sparsity + 0.1 * torch.sigmoid(input_variance - 1)
self.set_layer_sparsity(adaptive_sparsity)
""").body
                    
                    # Insert at beginning of forward method
                    node.body = adapt_code + node.body
                
                return self.generic_visit(node)
        
        # Transform and return
        transformer = WINAInjector()
        modified_tree = transformer.visit(tree)
        
        return ast.unparse(modified_tree)
    
    @staticmethod
    def add_importance_tracking(code: str) -> str:
        """Add neuron importance tracking for better sparsity decisions"""
        
        tracking_code = '''
class ImportanceTracker:
    """Track neuron importance over time for adaptive sparsity"""
    
    def __init__(self, size: int, momentum: float = 0.9):
        self.running_importance = torch.zeros(size)
        self.momentum = momentum
        
    def update(self, importance_scores: torch.Tensor):
        """Update running importance with exponential moving average"""
        self.running_importance = (
            self.momentum * self.running_importance + 
            (1 - self.momentum) * importance_scores.mean(dim=0)
        )
    
    def get_adaptive_mask(self, sparsity: float) -> torch.Tensor:
        """Get mask based on historical importance"""
        k = int((1 - sparsity) * len(self.running_importance))
        _, top_indices = torch.topk(self.running_importance, k)
        
        mask = torch.zeros_like(self.running_importance)
        mask[top_indices] = 1.0
        
        return mask

# Add to agent initialization
self.importance_trackers = {
    name: ImportanceTracker(module.in_features)
    for name, module in self.model.named_modules()
    if isinstance(module, WINALayer)
}
'''
        return code + tracking_code


if __name__ == "__main__":
    # Run demonstration
    demonstrate_wina_dgm_synergy()
    
    # Show code transformation example
    print("\n\n=== Code Transformation Example ===")
    
    example_code = '''
def forward(self, x):
    x = self.embedding(x)
    x = self.transformer(x)
    return self.output(x)
'''
    
    transformed = WINACodeTransformer.inject_dynamic_sparsity(example_code)
    print("Original:")
    print(example_code)
    print("\nTransformed with WINA:")
    print(transformed)