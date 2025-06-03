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
    performance: float = 0.0
    novelty: float = 0.0
    code: Optional[str] = None
    
    def __post_init__(self):
        self.wina = WINASelfOptimizer()
        self.wina.sparsity_config.update(self.config)

class DarwinGodelMachine:
    """
    Implements the Darwin-Gödel Machine for evolving WINA agents
    """
    
    def __init__(
        self,
        initial_agent: Agent,
        population_size: int = 20,
        novelty_weight: float = 0.3,
        mutation_rate: float = 0.1,
        elite_size: int = 2,
    ):
        self.population = [initial_agent]
        self.population_size = population_size
        self.novelty_weight = novelty_weight
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.history = []
        self.generation = 0
        
    def _initialize_population(self):
        """Initialize population with random variations"""
        while len(self.population) < self.population_size:
            new_agent = self._mutate_agent(random.choice(self.population))
            self.population.append(new_agent)
    
    def _mutate_agent(self, agent: Agent) -> Agent:
        """Create a mutated copy of an agent"""
        new_config = agent.config.copy()
        
        # Mutate global sparsity
        if random.random() < self.mutation_rate:
            new_config['global'] = np.clip(
                new_config['global'] + np.random.normal(0, 0.05),
                0.1, 0.9
            )
            
        # Mutate layer schedule
        if 'layer_schedule' in new_config and random.random() < self.mutation_rate:
            schedule = new_config['layer_schedule']
            idx = random.randint(0, len(schedule)-1)
            schedule[idx] = np.clip(
                schedule[idx] + np.random.normal(0, 0.05),
                0.1, 0.9
            )
            
        return Agent(
            id=f"agent_{len(self.population)}_{self.generation}",
            config=new_config
        )
    
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
            
            # Evaluate all agents
            for agent in self.population:
                metrics = evaluation_task(agent.wina)
                agent.performance = metrics.get('fitness', 0)
                agent.novelty = self._evaluate_novelty(agent, self.population)
            
            # Sort by combined fitness (performance + novelty)
            for agent in self.population:
                agent.fitness = (1 - self.novelty_weight) * agent.performance + \
                              self.novelty_weight * agent.novelty
            
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Save best agents
            self.history.append({
                'generation': gen,
                'best_fitness': self.population[0].fitness,
                'best_performance': self.population[0].performance,
                'avg_fitness': np.mean([a.fitness for a in self.population])
            })
            
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
