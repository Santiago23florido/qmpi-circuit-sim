# src/qsim/core/state.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def uniform_local_size(nqubits: int, nbp: int) -> int:
    N = 1 << nqubits
    if N % nbp != 0:
        raise ValueError(f"N=2^{nqubits}={N} is not divisible by nbp={nbp}")
    local_N = N // nbp
    if (nbp & (nbp - 1)) != 0:
        raise ValueError("nbp must be a power of two")
    if (local_N & (local_N - 1)) != 0:
        raise ValueError("local_N must be a power of two")
    return local_N


def zerostatevec_uniform(nqubits: int, rank: int, nbp: int, dtype=np.complex128) -> np.ndarray:
    local_N = uniform_local_size(nqubits, nbp)
    psi_local = np.zeros(local_N, dtype=dtype)
    if rank == 0:
        psi_local[0] = 1.0 + 0.0j
    return psi_local


@dataclass
class StateVector:
    nqubits: int
    psi: np.ndarray
    rank: int = 0
    size: int = 1

    @property
    def dim_local(self) -> int:
        return int(self.psi.size)

    @property
    def local_offset(self) -> int:
        return self.rank * self.dim_local