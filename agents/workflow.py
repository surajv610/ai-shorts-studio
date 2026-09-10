"""V1 orchestration contract. Provider implementations are intentionally separate.

This module re-exports the transition logic now owned by the Master Agent.
"""

from agents.master import ALLOWED_TRANSITIONS, PAUSE_STATES, InvalidTransitionError, can_transition

__all__ = [
    "ALLOWED_TRANSITIONS",
    "PAUSE_STATES",
    "InvalidTransitionError",
    "can_transition",
]
