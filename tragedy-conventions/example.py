"""
Simple example showing how to use the coordination game models.
"""

from src.models import NGameCoordinationSystem
from src.utilities import print_section_header

# Set random seed for reproducibility
import numpy as np
np.random.seed(42)

# Example 1: Single independent game (validation)
print_section_header("Example 1: Independent Game (n=1)")

model_single = NGameCoordinationSystem(
    n_games=1,
    alpha_coupling=0.0,
    constant=1.1,
    beta=1.0
)

results = model_single.measure_basins(n_trials=10000, seed=42)
print(f"All-A success: {results['All-A']:.1%}")
print(f"All-B success: {results['All-B']:.1%}")

# Example 2: Interdependent 2-game system
print_section_header("Example 2: Interdependent Games (n=2)")

model_inter = NGameCoordinationSystem(
    n_games=2,
    alpha_coupling=0.5,
    constant=0.6,
    beta=1.0
)

results = model_inter.measure_basins(n_trials=10000, seed=42)
print(f"All-A success: {results['All-A']:.1%}")
print(f"Fragmented: {results['Fragmented']:.1%}")
print(f"All-B success: {results['All-B']:.1%}")

# Example 3: Correlation effects
print_section_header("Example 3: Correlation Effects")

for r in [0.0, 0.5, 0.9]:
    model = NGameCoordinationSystem(
        n_games=2,
        alpha_coupling=2.0,
        constant=0.6,
        within_correlation=r
    )
    results = model.measure_basins(n_trials=2000, seed=42)
    print(f"Correlation r={r:.1f}: All-A={results['All-A']:.1%}")
