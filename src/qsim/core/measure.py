from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

from .state import StateVector


@dataclass(frozen=True)
class MeasurementResult:
    counts: Dict[str, int]
    probabilities: Optional[np.ndarray] = None  # full distribution if requested


def probabilities_all(state: StateVector) -> np.ndarray:
    p = np.abs(state.psi) ** 2
    # numerical cleanup
    s = float(p.sum())
    if s != 0.0:
        p = p / s
    return p


def sample_measurements(
    state: StateVector,
    shots: int,
    rng: np.random.Generator,
    return_probabilities: bool = False,
) -> MeasurementResult:
    """
    Sample bitstrings according to p[i] = |ψ[i]|^2.

    Output bitstrings are printed MSB->LSB (standard binary string),
    while qubit indices in gates use q=0 as LSB.
    """
    p = probabilities_all(state)
    n = state.nqubits
    outcomes = rng.choice(state.dim, size=shots, p=p)

    counts: Dict[str, int] = {}
    for x in outcomes:
        bitstr = format(int(x), f"0{n}b")  # MSB -> LSB
        counts[bitstr] = counts.get(bitstr, 0) + 1

    return MeasurementResult(counts=counts, probabilities=p if return_probabilities else None)