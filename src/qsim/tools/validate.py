from __future__ import annotations

import numpy as np
from typing import Tuple
from ..core.state import StateVector


def norm2(state: StateVector) -> float:
    return float(np.vdot(state.psi, state.psi).real)


def check_norm(state: StateVector, tol: float = 1e-10) -> Tuple[bool, float]:
    n2 = norm2(state)
    return (abs(n2 - 1.0) <= tol, n2)


def bell_expected_probabilities() -> np.ndarray:
    """
    For Bell state (|00> + |11>)/sqrt(2):
    P(00)=0.5, P(11)=0.5
    """
    p = np.zeros(4, dtype=np.float64)
    p[0] = 0.5
    p[3] = 0.5
    return p
