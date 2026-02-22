import json
from pathlib import Path
from .circuit_parser import Circuit


def save_circuit_json(circuit: Circuit, path: Path) -> None:
    path.write_text(json.dumps(circuit.to_dict(), indent=2), encoding="utf-8")


def load_circuit_json(path: Path) -> Circuit:
    data = json.loads(path.read_text(encoding="utf-8"))
    return Circuit.from_dict(data)