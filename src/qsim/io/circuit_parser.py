from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional


@dataclass(frozen=True)
class Operation:
    name: str
    targets: Tuple[int, ...]
    params: Tuple[float, ...] = ()


@dataclass(frozen=True)
class Circuit:
    nqubits: int
    ops: Tuple[Operation, ...]
    has_measure_all: bool = False

    def to_dict(self) -> dict:
        return {
            "nqubits": self.nqubits,
            "has_measure_all": self.has_measure_all,
            "ops": [
                {"name": op.name, "targets": list(op.targets), "params": list(op.params)}
                for op in self.ops
            ],
        }

    @staticmethod
    def from_dict(d: dict) -> "Circuit":
        ops = []
        for item in d["ops"]:
            ops.append(
                Operation(
                    name=str(item["name"]),
                    targets=tuple(int(x) for x in item["targets"]),
                    params=tuple(float(x) for x in item.get("params", [])),
                )
            )
        return Circuit(
            nqubits=int(d["nqubits"]),
            ops=tuple(ops),
            has_measure_all=bool(d.get("has_measure_all", False)),
        )


def _parse_line(line: str) -> Optional[List[str]]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    # allow inline comments
    if "#" in line:
        line = line.split("#", 1)[0].strip()
        if not line:
            return None
    return line.split()


def parse_circuit_txt(path: Path) -> Circuit:
    nqubits: Optional[int] = None
    ops: List[Operation] = []
    has_measure_all = False

    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = _parse_line(raw)
        if parts is None:
            continue

        head = parts[0].upper()
        if head == "NQUBITS":
            if len(parts) != 2:
                raise ValueError("Formato: nqubits N")
            nqubits = int(parts[1])
            continue

        if head == "MEASURE_ALL":
            has_measure_all = True
            ops.append(Operation("MEASURE_ALL", ()))
            continue

        if head in {"H", "X", "Z"}:
            if len(parts) != 2:
                raise ValueError(f"Formato: {head} q")
            q = int(parts[1])
            ops.append(Operation(head, (q,)))
            continue

        if head == "RZ":
            if len(parts) != 3:
                raise ValueError("Formato: RZ q theta")
            q = int(parts[1])
            theta = float(parts[2])
            ops.append(Operation("RZ", (q,), (theta,)))
            continue

        if head == "CNOT":
            if len(parts) != 3:
                raise ValueError("Formato: CNOT control target")
            c = int(parts[1])
            t = int(parts[2])
            ops.append(Operation("CNOT", (c, t)))
            continue

        raise ValueError(f"Operación no soportada: {head}")

    if nqubits is None:
        raise ValueError("Falta línea obligatoria: nqubits N")

    return Circuit(nqubits=nqubits, ops=tuple(ops), has_measure_all=has_measure_all)