"""Utility functions for analysis and visualization.

This module contains helper functions for analyzing coordination games
and evaluating welfare outcomes.
"""

import numpy as np


def print_section_header(title: str, description: str = "") -> None:
    """Print formatted section header for notebook organization."""
    print("\n" + "="*80)
    print(title.upper())
    print("="*80)
    if description:
        print(description)
        print()


class UtilityEvaluator:
    """
    Compute welfare/utility for outcomes in n-game coordination systems.
    
    For a single game at state x (proportion playing A):
        - A-players receive: α * x
        - B-players receive: β * (1-x)
        - Average utility: x(αx) + (1-x)(β(1-x)) = αx² + β(1-x)²
    
    For interdependent games:
        - α_i depends on states of neighboring games
        - Compute average utility for each game, then aggregate
    """
    
    @staticmethod
    def compute_game_utility(x: float, alpha: float, beta: float) -> float:
        """
        Compute average utility in a single game at state x.
        
        Parameters:
        -----------
        x : float
            Proportion playing A
        alpha : float
            Payoff parameter for A-coordination
        beta : float
            Payoff parameter for B-coordination
        
        Returns:
        --------
        float : Average utility per player
        """
        utility_A_players = alpha * x
        utility_B_players = beta * (1 - x)
        average_utility = x * utility_A_players + (1 - x) * utility_B_players
        return average_utility
    
    @staticmethod
    def compute_system_utility(x_final: np.ndarray, model) -> dict:
        """
        Compute utilities for a final state in n-game system.
        
        Parameters:
        -----------
        x_final : ndarray
            Final state vector
        model : NGameCoordinationSystem
            Model instance used for simulation
        
        Returns:
        --------
        dict with utility metrics
        """
        n = model.n
        game_utilities = np.zeros(n)
        
        for i in range(n):
            alpha_i = model.compute_alpha(x_final, i)
            game_utilities[i] = UtilityEvaluator.compute_game_utility(
                x_final[i], alpha_i, model.beta
            )
        
        total_utility = np.sum(game_utilities)
        average_utility = np.mean(game_utilities)
        
        # Compute benchmarks
        x_all_A = np.ones(n)
        all_A_utility = 0
        for i in range(n):
            alpha_i = model.compute_alpha(x_all_A, i)
            all_A_utility += UtilityEvaluator.compute_game_utility(1.0, alpha_i, model.beta)
        
        x_all_B = np.zeros(n)
        all_B_utility = 0
        for i in range(n):
            alpha_i = model.compute_alpha(x_all_B, i)
            all_B_utility += UtilityEvaluator.compute_game_utility(0.0, alpha_i, model.beta)
        
        # Compute proportion of potential gain achieved
        if all_A_utility > all_B_utility:
            utility_achieved = (total_utility - all_B_utility) / (all_A_utility - all_B_utility)
        else:
            utility_achieved = 0.0
        
        return {
            'total_utility': total_utility,
            'average_utility': average_utility,
            'game_utilities': game_utilities,
            'all_A_utility': all_A_utility,
            'all_B_utility': all_B_utility,
            'utility_achieved': utility_achieved
        }
