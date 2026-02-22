from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from .config import SimConfig
from .io.circuit_parser import parse_circuit_txt, Circuit
from .io.circuit_json import save_circuit_json, load_circuit_json
from .core.ops import simulate
from .core.measure import sample_measurements
from .tools.validate import check_norm
from .tools.benchmark import run_benchmarks
from mpi4py import MPI


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qsim", description="Serial state-vector quantum circuit simulator (NumPy).")
    p.add_argument("--circuit", type=str, default=None, help="Path to .txt circuit file")
    p.add_argument("--load-json", type=str, default=None, help="Load circuit from JSON file")
    p.add_argument("--save-json", type=str, default=None, help="Save loaded circuit as JSON")
    p.add_argument("--shots", type=int, default=1024, help="Number of measurement shots")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--check-norm", action="store_true", help="Check that ||psi||^2 ~= 1")
    p.add_argument("--benchmark", action="store_true", help="Run a small benchmark and exit")
    p.add_argument("--show-top", type=int, default=8, help="Show top-k measurement outcomes")
    return p


def _load_circuit(cfg: SimConfig) -> Circuit:
    if cfg.load_json_path is not None:
        circuit = load_circuit_json(cfg.load_json_path)
    elif cfg.circuit_path is not None:
        circuit = parse_circuit_txt(cfg.circuit_path)
    else:
        raise ValueError("Provide --circuit or --load-json")
    return circuit


def main(argv: Optional[list[str]] = None) -> int:
    com = MPI.COMM_WORLD.Dup()
    nbp= com.Get_size()
    rank = com.Get_rank()
    
    args = _build_argparser().parse_args(argv)

    cfg = SimConfig(
        circuit_path=Path(args.circuit) if args.circuit else None,
        load_json_path=Path(args.load_json) if args.load_json else None,
        save_json_path=Path(args.save_json) if args.save_json else None,
        shots=int(args.shots),
        seed=args.seed,
        check_norm=bool(args.check_norm),
        benchmark=bool(args.benchmark),
    )

    if cfg.benchmark:
        out = run_benchmarks(Path("results"), seed=cfg.seed)
        print(f"[benchmark] wrote: {out}")
        return 0

    circuit = _load_circuit(cfg)

    if cfg.save_json_path is not None:
        save_circuit_json(circuit, cfg.save_json_path)
        print(f"[json] saved circuit to: {cfg.save_json_path}")

    # Unitary evolution
    state = simulate(circuit)

    if cfg.check_norm:
        ok, n2 = check_norm(state)
        print(f"[norm] ||psi||^2 = {n2:.12f}  ok={ok}")

    # Measurement sampling
    rng = np.random.default_rng(cfg.seed)
    meas = sample_measurements(state, shots=cfg.shots, rng=rng, return_probabilities=False)

    # Print top-k
    items = sorted(meas.counts.items(), key=lambda kv: kv[1], reverse=True)
    topk = items[: max(1, int(args.show_top))]

    print(f"[shots] {cfg.shots}")
    for bitstr, c in topk:
        print(f"{bitstr} : {c}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())