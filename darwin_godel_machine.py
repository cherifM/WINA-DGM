"""
Darwin-Gödel Machine (DGM) for Evolutionary Optimization of WINA Agents
"""

import numpy as np
from typing import List, Dict, Tuple, Callable, Optional
import torch
import random
import copy
from dataclasses import dataclass
from main import WINASelfOptimizer

@dataclass
class Agent:
    """Base agent class for DGM evolution"""
    id: str
    config: Dict
    model: torch.nn.Module  # PyTorch model for this agent
    performance: float = 0.0
    novelty: float = 0.0
    code: Optional[str] = None
    
    def __post_init__(self):
        # Create a deep copy of the model to avoid sharing weights
        model_copy = copy.deepcopy(self.model)
        self.wina = WINASelfOptimizer(model=model_copy)
        if hasattr(self.wina, 'sparsity_config') and self.config:
            self.wina.sparsity_config.update(self.config)

class DarwinGodelMachine:
    """
    Implements the Darwin-Gödel Machine for evolving WINA agents
    """
    
    def __init__(
        self,
        initial_agent: Agent,
        population_size: int = 20,
        novelty_weight: float = 0.4,
        mutation_rate: float = 0.15,
        elite_size: int = 3,
        crossover_prob: float = 0.7,
        min_sparsity: float = 0.1,
        max_sparsity: float = 0.95,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.population = [initial_agent]
        self.population_size = population_size
        self.novelty_weight = novelty_weight
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.crossover_prob = crossover_prob
        self.min_sparsity = min_sparsity
        self.max_sparsity = max_sparsity
        self.device = device
        self.generation = 0
        self.history = []
        
        # Initialize population
        self._initialize_population(initial_agent)
    
    def _initialize_population(self, base_agent: Agent):
        """Initialize the population with random variations of the base agent."""
        import copy
        import numpy as np
        
        # Keep the original agent
        self.population = [base_agent]
        
        # Create variations
        while len(self.population) < self.population_size:
            new_agent = self._mutate_agent(base_agent)
            self.population.append(new_agent)
    
    def _mutate_agent(self, agent: Agent) -> Agent:
        """Create a mutated copy of an agent."""
        import copy
        import numpy as np
        
        # Create a deep copy of the config to avoid modifying the original
        new_config = copy.deepcopy(agent.config)
        
        # Mutate sparsity values with bounds checking
        for key in new_config:
            if isinstance(new_config[key], (int, float)):
                # Add multiplicative noise with bounds checking
                new_value = new_config[key] * np.random.normal(1.0, 0.2)
                new_config[key] = np.clip(
                    new_value,
                    self.min_sparsity,
                    self.max_sparsity
                )
        
        # Create a new agent with the same model and new config
        return Agent(
            id=f"agent_{len(self.population)}_{self.generation}",
            model=agent.model,  # Share the model reference
            config=new_config
        )
        
    def _crossover(self, parent1: Agent, parent2: Agent) -> Agent:
        """Create a new agent by crossing over two parents."""
        import random
        import numpy as np
        
        child_config = {}
        for key in parent1.config:
            if isinstance(parent1.config[key], (int, float)):
                # Blend parameters from both parents
                alpha = np.random.uniform(0.3, 0.7)  # Blend ratio
                child_config[key] = (
                    alpha * parent1.config[key] + 
                    (1 - alpha) * parent2.config[key]
                )
            else:
                # Copy other parameters from a random parent
                child_config[key] = random.choice([
                    parent1.config[key],
                    parent2.config[key]
                ])
        
        return Agent(
            id=f"agent_{len(self.population)}_{self.generation}",
            model=parent1.model,  # Use the model from first parent
            config=child_config
        )
        
    def _tournament_selection(self, tournament_size: int = 3) -> Agent:
        """Select an agent using tournament selection."""
        participants = random.sample(
            self.population, 
            min(tournament_size, len(self.population))
        )
        return max(participants, key=lambda x: x.fitness)
        
    def _select_and_reproduce(self):
        """Select parents and create the next generation."""
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Keep elites
        new_population = self.population[:self.elite_size]
        
        # Create next generation
        while len(new_population) < self.population_size:
            # Select parents using tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if (random.random() < self.crossover_prob and 
                len(new_population) < self.population_size - 1):
                child = self._crossover(parent1, parent2)
                new_population.append(child)
            
            # Mutation
            if len(new_population) < self.population_size:
                child = self._mutate_agent(random.choice([parent1, parent2]))
                new_population.append(child)
        
        self.population = new_population
        
    def _evaluate_population(self, fitness_function):
        """Evaluate all agents in the population."""
        for agent in self.population:
            if not hasattr(agent, 'fitness'):
                agent.fitness = fitness_function(agent)
    
    def evolve(self, fitness_function, n_generations: int = 10):
        """Evolve the population for a number of generations."""
        for gen in range(n_generations):
            self.generation = gen
            self._evaluate_population(fitness_function)
            self._select_and_reproduce()
            self.history.append(self.get_best_agent())
    
    def get_best_agent(self) -> Agent:
        """Get the best agent from the current population."""
        return max(self.population, key=lambda x: x.fitness)
    
    def get_average_fitness(self) -> float:
        """Calculate the average fitness of the population."""
        return sum(agent.fitness for agent in self.population) / len(self.population)
    
    def _evaluate_novelty(self, agent: Agent, population: List[Agent]) -> float:
        """Calculate novelty score based on behavior space distance"""
        if len(population) <= 1:
            return 1.0
            
        # Simple behavior characterization: sparsity config distances
        behaviors = [
            np.array([a.config['global']] + a.config.get('layer_schedule', []))
            for a in population if a != agent
        ]
        
        if not behaviors:
            return 1.0
            
        agent_behavior = np.array([agent.config['global']] + agent.config.get('layer_schedule', []))
        distances = [np.linalg.norm(agent_behavior - b) for b in behaviors]
        return np.mean(sorted(distances)[:3])  # Average distance to 3 nearest neighbors
    
    def _select_parents(self, population: List[Agent]) -> List[Agent]:
        """Select parents using tournament selection"""
        tournament_size = min(5, len(population) // 2)
        tournament = random.sample(population, tournament_size)
        return sorted(tournament, key=lambda x: x.performance, reverse=True)[:2]
    
    def _breed(self, parent1: Agent, parent2: Agent) -> Agent:
        """Create offspring by crossover and mutation"""
        # Uniform crossover
        child_config = {}
        for key in parent1.config:
            if random.random() < 0.5:
                child_config[key] = copy.deepcopy(parent1.config[key])
            else:
                child_config[key] = copy.deepcopy(parent2.config[key])
                
        child = Agent(
            id=f"agent_{len(self.population)}_{self.generation}",
            config=child_config
        )
        
        # Apply mutation
        return self._mutate_agent(child)
    
    def evolve_population(self, n_generations: int, evaluation_task: Callable):
        """Run the evolutionary process"""
        self._initialize_population()
        
        for gen in range(n_generations):
            self.generation = gen
            gen_fitnesses = []
            
            print(f"\n--- Generation {gen + 1}/{n_generations} ---")
            
            # Evaluate all agents
            for i, agent in enumerate(self.population):
                try:
                    # Evaluate agent
                    metrics = evaluation_task(agent.wina)
                    agent.performance = metrics.get('accuracy', 0)
                    agent.loss = metrics.get('loss', float('inf'))
                    
                    # Calculate fitness (weighted sum of performance and FLOPs reduction)
                    flops_reduction = metrics.get('flops_reduction', 0)
                    agent.fitness = agent.performance + 0.3 * flops_reduction
                    
                    # Evaluate novelty
                    agent.novelty = self._evaluate_novelty(agent, self.population)
                    
                    gen_fitnesses.append(agent.fitness)
                    
                    print(f"Agent {i+1}: Acc={agent.performance:.4f}, "
                          f"FLOPs↓={flops_reduction*100:.1f}%, "
                          f"Fitness={agent.fitness:.4f}")
                          
                except Exception as e:
                    print(f"Error evaluating agent {i+1}: {str(e)}")
                    agent.fitness = -float('inf')
                    agent.performance = 0
                    agent.novelty = 0
            
            # Sort by combined fitness (performance + novelty)
            for agent in self.population:
                agent.fitness = (1 - self.novelty_weight) * agent.performance + \
                              self.novelty_weight * agent.novelty
            
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Calculate statistics
            best_fitness = max(a.fitness for a in self.population)
            avg_fitness = np.mean([a.fitness for a in self.population])
            best_performance = max(a.performance for a in self.population)
            
            # Save generation statistics
            self.history.append({
                'generation': gen,
                'best_fitness': best_fitness,
                'best_performance': best_performance,
                'avg_fitness': avg_fitness,
                'best_flops_reduction': max(a.wina.sparsity_config['global'] for a in self.population)
            })
            
            print(f"\nGeneration {gen} Summary:")
            print(f"Best Fitness: {best_fitness:.4f}")
            print(f"Best Accuracy: {best_performance*100:.2f}%")
            print(f"Average Fitness: {avg_fitness:.4f}")
            print(f"Best Sparsity: {self.population[0].wina.sparsity_config['global']:.2f}")
            
            # Create next generation (elitism + offspring)
            next_generation = self.population[:self.elite_size]
            
            while len(next_generation) < self.population_size:
                parent1, parent2 = self._select_parents(self.population)
                child = self._breed(parent1, parent2)
                next_generation.append(child)
            
            self.population = next_generation[:self.population_size]
            
            print(f"Generation {gen}: Best Fitness = {self.history[-1]['best_fitness']:.4f}")
    
    def get_best_agent(self) -> Tuple[Agent, Dict]:
        """Return the best agent and its performance"""
        best_agent = max(self.population, key=lambda x: x.performance)
        return best_agent, {
            'performance': best_agent.performance,
            'config': best_agent.config,
            'generation': self.generation
        }
