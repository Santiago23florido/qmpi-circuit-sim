from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SimConfig:
    circuit_path: Optional[Path] = None
    load_json_path: Optional[Path] = None
    save_json_path: Optional[Path] = None

    shots: int = 1024
    seed: Optional[int] = None
    check_norm: bool = False
    benchmark: bool = False