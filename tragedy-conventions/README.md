# The Tragedy of Social Coordination

Code repository for "The Tragedy of Social Coordination" (submitted for review - anonymized).

## Overview

This repository contains Python code implementing evolutionary game theory simulations to investigate coordination dynamics in interdependent games. The research demonstrates that standard coordination mechanisms (correlation and conditional cooperation) that work reliably in independent games can backfire or fail when games become interdependent.

## Key Findings

- **Basin Shrinkage**: Interdependence dramatically reduces the viability of optimal coordination
- **Correlation Backfires**: In weakly coupled systems, correlation mechanisms reduce optimal outcomes
- **Conditional Cooperation Fragility**: Requires accurate threshold setting and no "weak links"
- **Scale Effects**: Coordination difficulty increases with system size
- **Welfare Analysis**: Fragmented outcomes can be worse than uniform suboptimal coordination

## Installation

```bash
git clone https://github.com/YOUR-USERNAME/tragedy-of-conventions.git
cd tragedy-of-conventions
pip install -r requirements.txt
```

## Repository Structure

```
tragedy-conventions/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── LICENSE                            # License file
├── src/                               # Core Python modules
│   ├── __init__.py                   # Package initialization
│   ├── models.py                     # Game theory models
│   └── utilities.py                  # Analysis utilities
├── notebooks/                         # Analysis notebooks
│   └── full_analysis.ipynb           # Complete analysis
└── figures/                           # Generated figures
```

## Quick Start

### Using the Jupyter Notebook

The easiest way to explore the code is through the Jupyter notebook:

```bash
jupyter notebook notebooks/Tragedy_Conventions_Notebook.ipynb
```

Run all cells to reproduce the paper's results and figures.

### Using the Python Modules

You can also import and use the models directly:

```python
from src.models import NGameCoordinationSystem

# Create a 2-game interdependent system
model = NGameCoordinationSystem(
    n_games=2,
    alpha_coupling=0.5,  # interdependence strength
    constant=0.6,        # independent value
    beta=1.0,           # B-coordination payoff
    within_correlation=0.0  # no correlation
)

# Measure basins of attraction
results = model.measure_basins(n_trials=10000)
print(f"All-A success rate: {results['All-A']:.1%}")
```

## Model Parameters

- **n_games**: Number of coordination games in the system
- **alpha_coupling (a)**: Interdependence strength (how much neighbors influence local payoffs)
- **constant (c)**: Independent value component
- **beta (β)**: Payoff for B-coordination (held constant at 1.0)
- **within_correlation (r)**: Degree of positive assortment within games (0 = random matching, 1 = perfect assortment)

## Key Model Features

### Interdependence Structure

In a ring network topology:
- **n=1**: α = c (independent game)
- **n=2**: α₁ = a·x₂ + c, α₂ = a·x₁ + c
- **n≥3**: αᵢ = a·(xᵢ₋₁ + xᵢ₊₁)/2 + c (periodic boundaries)

### Mechanisms Tested

1. **Correlated Strategies**: Positive assortment in matching (parameter r)
2. **Conditional Cooperation**: Strategy C that switches when commitment reaches threshold θ

## Reproducibility

All simulations use `RANDOM_SEED = 42` for reproducibility. Running the notebook should produce results identical to those reported in the paper.

### Package Versions

The code was developed and tested with:
- NumPy: 2.2.5
- Matplotlib: 3.8.4
- SciPy: 1.14.1

These versions are documented in the notebook output for full reproducibility.

## Example Analyses

### Basin Shrinkage (Block 4)

```python
# Compare independent vs interdependent 2-game system
model_indep = NGameCoordinationSystem(n_games=1, alpha_coupling=0.0, constant=0.85)
model_inter = NGameCoordinationSystem(n_games=2, alpha_coupling=0.5, constant=0.6)

results_indep = model_indep.measure_basins(n_trials=100000)
results_inter = model_inter.measure_basins(n_trials=100000)

print(f"Independent: {results_indep['All-A']:.1%}")
print(f"Interdependent: {results_inter['All-A']:.1%}")
```

### Correlation Effects (Block 5)

```python
# Test correlation effects
for r in [0.0, 0.5, 0.9]:
    model = NGameCoordinationSystem(
        n_games=2, alpha_coupling=0.5, constant=0.6, 
        within_correlation=r
    )
    results = model.measure_basins(n_trials=5000)
    print(f"r={r:.1f}: All-A={results['All-A']:.1%}")
```

### Scale Effects (Block 8)

```python
# Test how success varies with system size
for n in [2, 4, 6, 8, 10]:
    model = NGameCoordinationSystem(
        n_games=n, alpha_coupling=1.0, constant=0.3
    )
    results = model.measure_basins(n_trials=5000)
    print(f"n={n}: All-A={results['All-A']:.1%}")
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{anonymized2025tragedy,
  title={The Tragedy of Social Coordination},
  author={[Authors anonymized for review]},
  journal={[Under review]},
  year={2025},
  note={Code available at [repository URL]}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

[Contact information withheld during review process]

## Acknowledgments

[Acknowledgments withheld for anonymous review]
