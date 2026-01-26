"""Core models for interdependent coordination games.

This module implements evolutionary game theory simulations to investigate
coordination dynamics in interdependent games.
"""

import numpy as np
from scipy.integrate import odeint


# Simulation parameters (can be imported from notebooks)
SIMULATION_TIME = 50
SIMULATION_DT = 0.1
EQUILIBRIUM_THRESHOLD = 0.9


class NGameCoordinationSystem:
    """
    System of n coordination games with linear interdependence structure.
    
    Each game i is a coordination game where:
    - Strategy A yields: α_i · P(meet A-player in game i)
    - Strategy B yields: β · P(meet B-player in game i)
    
    The payoff parameter α_i depends linearly on neighboring games:
        α_i = a · (mean of neighbors' x values) + c
    
    where x_j is the proportion playing A in game j.
    
    Network Structure:
    - Ring topology: each game depends on its immediate neighbors
    - For n=1: α = c (independent game, no neighbors)
    - For n=2: α_1 = a·x_2 + c, α_2 = a·x_1 + c
    - For n≥3: α_i = a·(x_{i-1} + x_{i+1})/2 + c
    
    Parameters:
    -----------
    n_games : int
        Number of coordination games in the system
    alpha_coupling : float
        Interdependence strength parameter 'a' (a ≥ 0)
    constant : float
        Independent value parameter 'c' (c ≥ 0)
    beta : float
        Payoff for B-coordination (held constant, default β = 1)
    within_correlation : float
        Degree of positive assortment within each game, r ∈ [0,1]
    network_type : str
        Topology of interdependencies (default: 'ring')
    """
    
    def __init__(self, 
                 n_games: int,
                 alpha_coupling: float,
                 constant: float,
                 beta: float = 1.0,
                 within_correlation: float = 0.0,
                 network_type: str = 'ring'):
        
        # Validation
        if n_games < 1:
            raise ValueError("Must have at least 1 game")
        if alpha_coupling < 0:
            raise ValueError("Alpha coupling must be non-negative")
        if constant < 0:
            raise ValueError("Constant must be non-negative")
        if beta <= 0:
            raise ValueError("Beta must be positive")
        if not 0 <= within_correlation <= 1:
            raise ValueError("Within-game correlation must be in [0,1]")
        if network_type != 'ring':
            raise ValueError("Only 'ring' topology currently supported")
        
        self.n = n_games
        self.a = alpha_coupling
        self.c = constant
        self.beta = beta
        self.r = within_correlation
        self.network_type = network_type
    
    def compute_alpha(self, x: np.ndarray, game_idx: int) -> float:
        """
        Compute α_i for game i given current state vector x.
        
        For ring topology:
        - n=1: α = c (no neighbors)
        - n=2: α_i = a·x_j + c (where j is the other game)
        - n≥3: α_i = a·(x_{i-1} + x_{i+1})/2 + c (periodic boundaries)
        """
        if self.n == 1:
            return self.c
        elif self.n == 2:
            other_idx = 1 - game_idx
            return self.a * x[other_idx] + self.c
        else:
            left_neighbor = x[(game_idx - 1) % self.n]
            right_neighbor = x[(game_idx + 1) % self.n]
            return self.a * (left_neighbor + right_neighbor) / 2 + self.c
    
    def fitness_A(self, x: np.ndarray, game_idx: int) -> float:
        """
        Expected fitness of strategy A in game i.
        
        With correlation r, an A-player in game i meets another A-player
        with probability: r + (1-r)·x_i
        """
        alpha_i = self.compute_alpha(x, game_idx)
        x_i = x[game_idx]
        prob_meet_A = self.r + (1 - self.r) * x_i
        return alpha_i * prob_meet_A
    
    def fitness_B(self, x: np.ndarray, game_idx: int) -> float:
        """
        Expected fitness of strategy B in game i.
        
        With correlation r, a B-player in game i meets another B-player
        with probability: r + (1-r)·(1 - x_i)
        """
        x_i = x[game_idx]
        prob_meet_B = self.r + (1 - self.r) * (1 - x_i)
        return self.beta * prob_meet_B
    
    def replicator_dynamics(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Coupled replicator dynamics for all n games.
        
        For each game i:
            dx_i/dt = x_i(1 - x_i)(f_A^i - f_B^i)
        
        Parameters:
        -----------
        x : ndarray of length n
            Current state vector (proportion playing A in each game)
        t : float
            Time (required by odeint interface, not used)
        
        Returns:
        --------
        dx : ndarray of length n
            Time derivatives for all games
        """
        dx = np.zeros(self.n)
        
        for i in range(self.n):
            x_i = x[i]
            f_A = self.fitness_A(x, i)
            f_B = self.fitness_B(x, i)
            dx[i] = x_i * (1 - x_i) * (f_A - f_B)
        
        return dx
    
    def simulate(self, x0: np.ndarray, T: float = SIMULATION_TIME, 
                 dt: float = SIMULATION_DT) -> tuple:
        """
        Simulate coupled replicator dynamics from initial state.
        
        Parameters:
        -----------
        x0 : ndarray of length n
            Initial state (proportion playing A in each game)
        T : float
            Total simulation time
        dt : float
            Time step for output
        
        Returns:
        --------
        t : ndarray
            Time points
        x : ndarray of shape (len(t), n)
            State trajectory
        """
        if len(x0) != self.n:
            raise ValueError(f"Initial state must have length {self.n}")
        if not np.all((x0 >= 0) & (x0 <= 1)):
            raise ValueError("Initial state must have all values in [0,1]")
        
        t = np.arange(0, T, dt)
        x = odeint(self.replicator_dynamics, x0, t)
        return t, x
    
    def classify_outcome(self, x_final: np.ndarray, 
                        threshold: float = EQUILIBRIUM_THRESHOLD) -> str:
        """
        Classify final system state.
        
        Returns:
        --------
        'All-A' : All games converged to A
        'All-B' : All games converged to B
        'Fragmented' : Some games at A, others at B
        'Mixed' : At least one game in mixed equilibrium
        """
        classifications = []
        for x_i in x_final:
            if x_i > threshold:
                classifications.append('A')
            elif x_i < (1 - threshold):
                classifications.append('B')
            else:
                classifications.append('Mixed')
        
        if all(c == 'Mixed' for c in classifications):
            return 'Mixed'
        if 'Mixed' in classifications:
            return 'Mixed'
        if all(c == 'A' for c in classifications):
            return 'All-A'
        if all(c == 'B' for c in classifications):
            return 'All-B'
        return 'Fragmented'
    
    def measure_basins(self, n_trials: int = 10000, 
                       seed: int = None) -> dict:
        """
        Empirically measure basins of attraction via Monte Carlo sampling.
        
        Parameters:
        -----------
        n_trials : int
            Number of random initial conditions to test
        seed : int or None
            Random seed for reproducibility
        
        Returns:
        --------
        dict with keys:
            'All-A': proportion converging to All-A
            'All-B': proportion converging to All-B  
            'Fragmented': proportion with mixed outcomes across games
            'Mixed': proportion with mixed equilibria within games
            'n_trials': number of trials run
        """
        if seed is not None:
            np.random.seed(seed)
        
        outcomes = {'All-A': 0, 'All-B': 0, 'Fragmented': 0, 'Mixed': 0}
        
        for _ in range(n_trials):
            x0 = np.random.uniform(0, 1, self.n)
            t, x = self.simulate(x0)
            outcome = self.classify_outcome(x[-1])
            outcomes[outcome] += 1
        
        # Convert to proportions
        results = {k: v / n_trials for k, v in outcomes.items()}
        results['n_trials'] = n_trials
        
        return results


class NGameWithConditionalCooperation(NGameCoordinationSystem):
    """
    Extension of n-game system with conditional cooperation strategy.
    
    Adds a third strategy C (conditional cooperation) that follows the rule:
    "Play A when total commitment (A + C players) reaches threshold θ, 
     otherwise play B"
    
    State representation:
    - For each game i, track: x_A[i], x_C[i]
    - Proportion playing B: x_B[i] = 1 - x_A[i] - x_C[i]
    
    Parameters:
    -----------
    threshold : float or array-like
        Commitment threshold for triggering C-players' switch
    """
    
    def __init__(self,
                 n_games: int,
                 alpha_coupling: float,
                 constant: float,
                 beta: float = 1.0,
                 within_correlation: float = 0.0,
                 network_type: str = 'ring',
                 threshold: float = 0.5):
        
        super().__init__(n_games, alpha_coupling, constant, beta, 
                        within_correlation, network_type)
        
        # Handle threshold parameter
        if np.isscalar(threshold):
            self.theta = np.full(n_games, threshold)
        else:
            if len(threshold) != n_games:
                raise ValueError(f"Threshold array must have length {n_games}")
            self.theta = np.array(threshold)
        
        if not np.all((self.theta >= 0) & (self.theta <= 1)):
            raise ValueError("Thresholds must be in [0,1]")
    
    def compute_effective_x(self, x_A: np.ndarray, x_C: np.ndarray) -> np.ndarray:
        """
        Compute effective proportion playing A in each game.
        
        C-players switch to A when total commitment (x_A + x_C) >= θ
        """
        x_effective = np.zeros(self.n)
        
        for i in range(self.n):
            total_commitment = x_A[i] + x_C[i]
            if total_commitment >= self.theta[i]:
                x_effective[i] = total_commitment
            else:
                x_effective[i] = x_A[i]
        
        return x_effective
    
    def replicator_dynamics_with_C(self, state: np.ndarray, t: float) -> np.ndarray:
        """
        Replicator dynamics for 3-strategy system (A, B, C).
        
        State vector: [x_A[0], x_C[0], x_A[1], x_C[1], ..., x_A[n-1], x_C[n-1]]
        """
        x_A = state[0::2]
        x_C = state[1::2]
        x_B = 1 - x_A - x_C
        
        x_effective = self.compute_effective_x(x_A, x_C)
        
        dx_A = np.zeros(self.n)
        dx_C = np.zeros(self.n)
        
        for i in range(self.n):
            alpha_i = self.compute_alpha(x_effective, i)
            
            prob_meet_A = self.r + (1 - self.r) * x_effective[i]
            f_A = alpha_i * prob_meet_A
            
            if x_A[i] + x_C[i] < self.theta[i]:
                x_B_effective = x_B[i] + x_C[i]
            else:
                x_B_effective = x_B[i]
            
            prob_meet_B = self.r + (1 - self.r) * x_B_effective
            f_B = self.beta * prob_meet_B
            
            if x_A[i] + x_C[i] >= self.theta[i]:
                f_C = f_A
            else:
                f_C = f_B
            
            f_avg = x_A[i] * f_A + x_C[i] * f_C + x_B[i] * f_B
            
            dx_A[i] = x_A[i] * (f_A - f_avg)
            dx_C[i] = x_C[i] * (f_C - f_avg)
        
        d_state = np.zeros(2 * self.n)
        d_state[0::2] = dx_A
        d_state[1::2] = dx_C
        
        return d_state
    
    def simulate_with_C(self, x_A0: np.ndarray, x_C0: np.ndarray,
                       T: float = SIMULATION_TIME,
                       dt: float = SIMULATION_DT) -> tuple:
        """
        Simulate system with conditional cooperation.
        """
        if len(x_A0) != self.n or len(x_C0) != self.n:
            raise ValueError(f"Initial states must have length {self.n}")
        
        if not np.all((x_A0 >= 0) & (x_A0 <= 1)):
            raise ValueError("Initial x_A must be in [0,1]")
        if not np.all((x_C0 >= 0) & (x_C0 <= 1)):
            raise ValueError("Initial x_C must be in [0,1]")
        if not np.all(x_A0 + x_C0 <= 1):
            raise ValueError("x_A + x_C must not exceed 1")
        
        state0 = np.zeros(2 * self.n)
        state0[0::2] = x_A0
        state0[1::2] = x_C0
        
        t = np.arange(0, T, dt)
        state = odeint(self.replicator_dynamics_with_C, state0, t)
        
        x_A = state[:, 0::2]
        x_C = state[:, 1::2]
        
        return t, x_A, x_C
    
    def measure_basins_with_C(self, x_A_range: tuple = (0, 0.2),
                              x_C_range: tuple = (0, 0.8),
                              n_trials: int = 1000,
                              seed: int = None) -> dict:
        """
        Measure success rates with conditional cooperation.
        """
        if seed is not None:
            np.random.seed(seed)
        
        outcomes = {'All-A': 0, 'All-B': 0, 'Fragmented': 0, 'Mixed': 0}
        
        for _ in range(n_trials):
            x_A0 = np.random.uniform(x_A_range[0], x_A_range[1], self.n)
            x_C0 = np.random.uniform(x_C_range[0], x_C_range[1], self.n)
            
            total = x_A0 + x_C0
            if np.any(total > 1):
                scale = 0.99 / total.max()
                x_A0 *= scale
                x_C0 *= scale
            
            t, x_A, x_C = self.simulate_with_C(x_A0, x_C0)
            
            x_final_effective = self.compute_effective_x(x_A[-1], x_C[-1])
            outcome = self.classify_outcome(x_final_effective)
            outcomes[outcome] += 1
        
        results = {k: v / n_trials for k, v in outcomes.items()}
        results['n_trials'] = n_trials
        
        return results


class NGameWithConditionalCooperationHeterogeneous(NGameWithConditionalCooperation):
    """Extension to handle heterogeneous constant values across games."""
    
    def __init__(self,
                 n_games: int,
                 alpha_coupling: float,
                 constant: np.ndarray,
                 beta: float = 1.0,
                 within_correlation: float = 0.0,
                 network_type: str = 'ring',
                 threshold: float = 0.5):
        
        if len(constant) != n_games:
            raise ValueError(f"Constant array must have length {n_games}")
        self.c_array = np.array(constant)
        
        NGameCoordinationSystem.__init__(
            self, n_games, alpha_coupling, 0.0, beta, 
            within_correlation, network_type
        )
        
        if np.isscalar(threshold):
            self.theta = np.full(n_games, threshold)
        else:
            if len(threshold) != n_games:
                raise ValueError(f"Threshold array must have length {n_games}")
            self.theta = np.array(threshold)
    
    def compute_alpha(self, x: np.ndarray, game_idx: int) -> float:
        """Override to use heterogeneous constants."""
        if self.n == 1:
            return self.c_array[game_idx]
        elif self.n == 2:
            other_idx = 1 - game_idx
            return self.a * x[other_idx] + self.c_array[game_idx]
        else:
            left_neighbor = x[(game_idx - 1) % self.n]
            right_neighbor = x[(game_idx + 1) % self.n]
            return self.a * (left_neighbor + right_neighbor) / 2 + self.c_array[game_idx]
