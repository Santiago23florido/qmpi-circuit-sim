# qmpi-circuit-sim-serial

Serial (single-process) quantum circuit **state-vector** simulator in Python/NumPy.
Designed to be **MPI-ready** later by keeping a clean separation between:
- parsing / IO
- quantum core (state + gates + operations)
- measurement
- validation + benchmarks

## What is here (and physics)
This project implements a state-vector simulator: the quantum system is represented as a complex vector `psi` with `2^n` amplitudes. Gates are unitary transformations that mix amplitudes while preserving the norm (total probability = 1). Measurement uses `|psi[i]|^2` as a probability distribution.

## Physics in one paragraph
An `n`-qubit pure quantum state is a complex vector **psi** of length `2^n`. Each basis state
(e.g., `|00...0>`, `|00...1>`, ..., `|11...1>`) has an amplitude `psi[i]`. Measurement probabilities are
`p[i] = |psi[i]|^2`, and `sum(p) = 1`. Quantum gates are **unitary** operations that transform the
state while preserving the norm (total probability).

## Install
Create a virtual env (recommended) and install requirements:

```bash
pip install -r requirements.txt
```

## Run (important: set PYTHONPATH)
This repo uses the `src/` layout. Run with:

```bash
export PYTHONPATH=src
python -m qmpi.cli --circuit circuits/bell.txt --shots 1024 --seed 1 --check-norm
```

Or use `run.sh`.

## Circuit format (mini)
Example:

```text
nqubits 2
H 0
CNOT 0 1
MEASURE_ALL
```

Supported operations:
- 1-qubit: `H q`, `X q`, `Z q`, `RZ q theta`
- 2-qubit: `CNOT control target`
- measurement marker: `MEASURE_ALL` (if missing, we still measure at the end)

Qubit indexing convention:
- `q = 0` is the least significant bit (LSB) of the computational basis index.
- When printing bitstrings, we print MSB->LSB (standard binary string).

## JSON serialization
You can save/load circuits as JSON:

```bash
export PYTHONPATH=src
python -m qmpi.cli --circuit circuits/bell.txt --save-json circuits/bell.json
python -m qmpi.cli --load-json circuits/bell.json --shots 1024
```

## Tests
```bash
export PYTHONPATH=src
python -m unittest -v
```

## Benchmarks
```bash
export PYTHONPATH=src
python -m qmpi.cli --benchmark --seed 1
```

## Notes for later MPI
This version is intentionally serial. To parallelize with MPI later, you will:
- distribute the state vector `psi` across ranks
- replace global operations with local chunks + communication

The modular structure here is meant to make that straightforward.

## requirements.txt
Only NumPy is required because the physical/mathematical core is linear algebra on complex vectors.

```txt
numpy
```
