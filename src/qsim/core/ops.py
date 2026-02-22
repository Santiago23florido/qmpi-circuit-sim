# src/qsim/core/ops_mpi.py  (or replace your ops.py content with this)
from __future__ import annotations

import math
import numpy as np
from typing import Tuple
from mpi4py import MPI

from .state import StateVector, uniform_local_size, zerostatevec_uniform
from ..io.circuit_parser import Circuit, Operation
from . import gates as g


def apply_1q_gate_local_inplace(psi: np.ndarray, q: int, U: np.ndarray) -> None:
    if psi.ndim != 1:
        raise ValueError("psi must be a 1D vector")
    if U.shape != (2, 2):
        raise ValueError("U must be a 2x2 matrix")

    n = psi.size
    stride = 1 << q
    step = stride << 1
    if step > n:
        raise ValueError(f"Invalid qubit index q={q} for local state size n={n}")

    view = psi.reshape(-1, step)
    a = view[:, :stride].copy()
    b = view[:, stride:step].copy()
    view[:, :stride] = U[0, 0] * a + U[0, 1] * b
    view[:, stride:step] = U[1, 0] * a + U[1, 1] * b


def apply_1q_gate_mpi(psi_local: np.ndarray, q: int, U: np.ndarray, nqubits: int, comm: MPI.Comm) -> None:
    rank = comm.Get_rank()
    nbp = comm.Get_size()

    if U.shape != (2, 2):
        raise ValueError("U must be a 2x2 matrix")
    if not (0 <= q < nqubits):
        raise ValueError(f"q must be in [0, {nqubits-1}]")

    local_N = uniform_local_size(nqubits, nbp)
    if psi_local.size != local_N:
        raise ValueError("psi_local size does not match expected local_N")

    k = int(math.log2(local_N))

    if q < k:
        apply_1q_gate_local_inplace(psi_local, q, U)
        return

    d = q - k
    partner = rank ^ (1 << d)

    recv = np.empty_like(psi_local)
    comm.Sendrecv(sendbuf=psi_local, dest=partner, sendtag=q, recvbuf=recv, source=partner, recvtag=q)

    my_bit = (rank >> d) & 1
    if my_bit == 0:
        a = psi_local.copy()
        b = recv
        psi_local[:] = U[0, 0] * a + U[0, 1] * b
    else:
        a = recv
        b = psi_local.copy()
        psi_local[:] = U[1, 0] * a + U[1, 1] * b


def apply_cnot_local_inplace(psi: np.ndarray, control: int, target: int) -> None:
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


def simulate_mpi(circuit: Circuit, comm: MPI.Comm = MPI.COMM_WORLD) -> StateVector:
    rank = comm.Get_rank()
    nbp = comm.Get_size()

    nqubits = getattr(circuit, "nqubits", getattr(circuit, "qubits"))
    psi_local = zerostatevec_uniform(nqubits, rank, nbp)

    local_N = psi_local.size
    k_local = int(math.log2(local_N))

    for op in circuit.ops:
        kind, U = _op_to_unitary(op)

        if kind == "MEASURE_ALL":
            break

        if kind == "1Q":
            q = int(op.targets[0])
            apply_1q_gate_mpi(psi_local, q, U, nqubits, comm)

        elif kind == "CNOT":
            c, t = map(int, op.targets)
            if c < k_local and t < k_local:
                apply_cnot_local_inplace(psi_local, c, t)
            else:
                raise NotImplementedError("Distributed CNOT is not implemented yet")

        else:
            raise RuntimeError("Unexpected op kind")

    return StateVector(nqubits=nqubits, psi=psi_local, rank=rank, size=nbp)


def simulate(rank: int, nbp: int, com: MPI.Comm, circuit: Circuit) -> StateVector:
    if rank != com.Get_rank() or nbp != com.Get_size():
        raise ValueError("rank/nbp do not match the provided communicator")
    return simulate_mpi(circuit, com)