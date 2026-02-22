from __future__ import annotations

import numpy as np
from typing import Tuple
from .state import StateVector
from ..io.circuit_parser import Circuit, Operation
from . import gates as g


def apply_1q_gate_inplace(psi: np.ndarray, q: int, U: np.ndarray) -> None:
    """
    Apply a 1-qubit gate U on qubit q (q=0 is LSB) in-place.

    Vectorized strategy:
    - group ψ into blocks of size 2^(q+1)
    - within each block, the first half corresponds to bit q = 0, second half to bit q = 1
    - apply the 2x2 mixing to all blocks at once using reshape + slicing
    """
    if psi.ndim != 1:
        raise ValueError("psi must be a 1D vector")
    if U.shape != (2, 2):
        raise ValueError("U must be 2x2")

    n = psi.size
    stride = 1 << q
    step = stride << 1
    if step > n:
        raise ValueError(f"Invalid qubit index q={q} for state dimension {n}")

    view = psi.reshape(-1, step)  # shape (n/step, step)
    a = view[:, :stride].copy()
    b = view[:, stride:step].copy()

    view[:, :stride] = U[0, 0] * a + U[0, 1] * b
    view[:, stride:step] = U[1, 0] * a + U[1, 1] * b


def apply_cnot_inplace(psi: np.ndarray, control: int, target: int) -> None:
    """
    Apply CNOT(control -> target) in-place.

    Meaning:
    - if control bit is 1, flip the target bit.
    - This swaps amplitudes between basis states that differ only in target bit,
      but only in the subspace where control bit = 1.

    Vectorized approach using index masks:
    - find indices i where control=1 and target=0
    - swap ψ[i] <-> ψ[i xor 2^target]
    """
    if control == target:
        raise ValueError("control and target must be different")
    n = psi.size
    idx = np.arange(n, dtype=np.int64)

    c1 = ((idx >> control) & 1) == 1
    t0 = ((idx >> target) & 1) == 0
    mask = c1 & t0

    i = idx[mask]
    j = i ^ (1 << target)

    tmp = psi[i].copy()
    psi[i] = psi[j]
    psi[j] = tmp


def _op_to_unitary(op: Operation) -> Tuple[str, np.ndarray | None]:
    name = op.name.upper()
    if name == "H":
        return "1Q", g.H()
    if name == "X":
        return "1Q", g.X()
    if name == "Z":
        return "1Q", g.Z()
    if name == "RZ":
        theta = float(op.params[0])
        return "1Q", g.RZ(theta)
    if name == "CNOT":
        return "CNOT", None
    if name == "MEASURE_ALL":
        return "MEASURE_ALL", None
    raise ValueError(f"Unsupported op: {op.name}")


def simulate(circuit: Circuit) -> StateVector:
    """
    Run unitary part of the circuit and return final state vector.

    If MEASURE_ALL is present, we stop evolution when we reach it (so measurement happens after).
    If MEASURE_ALL is missing, we still return the final state after all ops.
    """
    state = StateVector.zero(circuit.nqubits)

    for op in circuit.ops:
        kind, U = _op_to_unitary(op)
        if kind == "MEASURE_ALL":
            break
        if kind == "1Q":
            q = op.targets[0]
            apply_1q_gate_inplace(state.psi, q, U)  # in-place
        elif kind == "CNOT":
            c, t = op.targets
            apply_cnot_inplace(state.psi, c, t)
        else:
            raise RuntimeError("Unexpected op kind")

    return state