from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

from ..io.circuit_parser import Circuit, Operation
from ..core.ops import simulate
from ..tools.validate import check_norm


@dataclass(frozen=True)
class BenchCase:
    nqubits: int
    depth: int


def random_circuit(nqubits: int, depth: int, rng: np.random.Generator) -> Circuit:
    """
    Create a simple random circuit for timing.
    We use only gates supported in the serial core.
    """
    ops: List[Operation] = []
    oneq = ["H", "X", "Z", "RZ"]
    for _ in range(depth):
        if rng.random() < 0.75:
            name = rng.choice(oneq)
            q = int(rng.integers(0, nqubits))
            if name == "RZ":
                theta = float(rng.random() * 2.0 * np.pi)
                ops.append(Operation("RZ", (q,), (theta,)))
            else:
                ops.append(Operation(str(name), (q,)))
        else:
            # CNOT
            c = int(rng.integers(0, nqubits))
            t = int(rng.integers(0, nqubits - 1))
            if t >= c:
                t += 1
            ops.append(Operation("CNOT", (c, t)))

    # marker
    ops.append(Operation("MEASURE_ALL", ()))
    return Circuit(nqubits=nqubits, ops=tuple(ops), has_measure_all=True)


def run_benchmarks(
    out_dir: Path,
    seed: Optional[int] = None,
    cases: Optional[List[BenchCase]] = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_serial.csv"

    rng = np.random.default_rng(seed)
    if cases is None:
        cases = [
            BenchCase(nqubits=18, depth=200),
            BenchCase(nqubits=20, depth=200),
            BenchCase(nqubits=22, depth=200),
        ]

    rows: List[dict] = []
    for case in cases:
        circuit = random_circuit(case.nqubits, case.depth, rng)

        t0 = time.perf_counter()
        state = simulate(circuit)
        t1 = time.perf_counter()

        ok, n2 = check_norm(state, tol=1e-8)
        rows.append(
            {
                "nqubits": case.nqubits,
                "depth": case.depth,
                "time_sec": (t1 - t0),
                "norm2": n2,
                "norm_ok": int(ok),
            }
        )

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    return out_path
    