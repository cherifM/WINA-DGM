# WINA-DGM: Weight-Importance-based Neural Architecture with Darwin-Gödel Machine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

WINA-DGM is a novel framework that combines Weight-Importance-based Neural Architecture (WINA) with Darwin-Gödel Machine (DGM) to create self-optimizing neural networks. The system automatically evolves neural network architectures by optimizing both performance and computational efficiency through dynamic sparsity patterns.

## 🌟 Key Features

- **Dynamic Sparsity**: Implements WINA for adaptive weight pruning based on importance scores
- **Evolutionary Optimization**: Uses DGM to evolve network architectures over generations
- **Theoretical Guarantees**: Incorporates error bounds and stability analysis
- **Efficient Training**: Reduces FLOPs while maintaining model accuracy
- **Self-Improving**: Agents optimize their own sparsity configurations

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/cherifM/WINA-DGM.git
cd WINA-DGM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

Run the example to see WINA-DGM in action:
```bash
python example_usage.py
```

This will train a simple neural network while evolving its sparsity configuration over multiple generations.

## 🧠 Core Components

### 1. WINA (Weight-Importance-based Neural Architecture)
- Implements SVD-based weight orthogonalization
- Computes importance scores for weights
- Applies dynamic sparsity masks
- Tracks theoretical error bounds

### 2. DGM (Darwin-Gödel Machine)
- Population-based evolution
- Novelty-weighted selection
- Performance caching
- Self-modifying code generation

## 📊 Example Results

```
Generation 0: Best Fitness = 0.7423
Generation 1: Best Fitness = 0.7689
Generation 2: Best Fitness = 0.7821
...
Generation 9: Best Fitness = 0.8456

Evolution complete!
Best accuracy: 0.8560
FLOPs reduction: 48.75%
```

## 🛠️ Customization

### Custom Task Integration

To use WINA-DGM with your own model and task:

1. Create an evaluation function that takes a WINA agent and returns performance metrics:

```python
def evaluate_my_task(wina_agent):
    # Your model and data loading code here
    model = MyModel()
    
    # Apply WINA masking to your model
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weights = module.weight.data
            mask, _ = wina_agent.compute_wina_mask(
                torch.randn(1, weights.shape[1]), 
                weights,
                wina_agent.sparsity_config['global']
            )
            module.weight.data = nn.Parameter(weights * mask)
    
    # Evaluate and return metrics
    accuracy = evaluate_model(model, test_loader)
    return {'accuracy': accuracy, 'fitness': accuracy}
```

2. Initialize and run the DGM:

```python
initial_agent = Agent(
    id="my_agent",
    config={
        'global': 0.5,
        'layer_schedule': [0.4, 0.5, 0.6],
        'importance_threshold': 0.1
    }
)

dgm = DarwinGodelMachine(
    initial_agent=initial_agent,
    population_size=20,
    novelty_weight=0.3
)

dgm.evolve_population(
    n_generations=10,
    evaluation_task=evaluate_my_task
)
```

## 📈 Performance Tuning

Key parameters to adjust:

- `population_size`: Larger values explore more solutions but are slower (default: 20)
- `novelty_weight`: Balances performance vs. exploration (default: 0.3)
- `mutation_rate`: Controls exploration rate (default: 0.1)
- `elite_size`: Number of top performers preserved between generations (default: 2)

## 📚 References

1. Original WINA Paper: [Link to Paper]
2. Darwin-Gödel Machines: [Link to Paper]
3. Sparse Neural Networks: [Link to Paper]

## 📄 Copyright and License

**Copyright © 2025 Cherif Mihoubi**  
**Email:** [cmihoubi@gmail.com](mailto:cmihoubi@gmail.com)

### Usage Restrictions

This software is provided for **academic and research purposes only**. Any commercial or industrial use is **strictly prohibited** without explicit written permission from the copyright holder.

**You are NOT permitted to:**

- Use this software for any commercial or industrial purposes
- Redistribute this software without permission
- Use this software in any product or service that is sold or generates revenue

For licensing inquiries, please contact the copyright holder at [cmihoubi@gmail.com](mailto:cmihoubi@gmail.com).

## 🤝 Contributing

Contributions are welcome for non-commercial purposes. Please submit a Pull Request for consideration.

## 📧 Contact

For questions or feedback, please contact:

- Cherif Mihoubi
- Email: [cmihoubi@gmail.com](mailto:cmihoubi@gmail.com)
