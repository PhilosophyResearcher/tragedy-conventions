"""
Tragedy of Conventions - Core package for interdependent coordination games.
"""

from .models import (
    NGameCoordinationSystem,
    NGameWithConditionalCooperation,
    NGameWithConditionalCooperationHeterogeneous,
    SIMULATION_TIME,
    SIMULATION_DT,
    EQUILIBRIUM_THRESHOLD
)

from .utilities import (
    UtilityEvaluator,
    print_section_header
)

__all__ = [
    'NGameCoordinationSystem',
    'NGameWithConditionalCooperation',
    'NGameWithConditionalCooperationHeterogeneous',
    'UtilityEvaluator',
    'print_section_header',
    'SIMULATION_TIME',
    'SIMULATION_DT',
    'EQUILIBRIUM_THRESHOLD'
]
