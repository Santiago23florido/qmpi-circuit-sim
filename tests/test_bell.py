import sys
from pathlib import Path
import unittest
import numpy as np

# Ensure src/ is importable when running tests directly
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from qsim.io.circuit_parser import Circuit, Operation
from qsim.core.ops import simulate
from qsim.core.measure import probabilities_all


class TestBell(unittest.TestCase):
    def test_bell_probabilities(self):
        # Bell circuit: H 0; CNOT 0 1
        circuit = Circuit(
            nqubits=2,
            ops=(
                Operation("H", (0,)),
                Operation("CNOT", (0, 1)),
                Operation("MEASURE_ALL", ()),
            ),
            has_measure_all=True,
        )
        state = simulate(circuit)
        p = probabilities_all(state)

        self.assertTrue(np.isclose(p.sum(), 1.0, atol=1e-10))
        self.assertTrue(np.isclose(p[0], 0.5, atol=1e-8))  # |00>
        self.assertTrue(np.isclose(p[3], 0.5, atol=1e-8))  # |11>
        self.assertTrue(np.isclose(p[1], 0.0, atol=1e-8))
        self.assertTrue(np.isclose(p[2], 0.0, atol=1e-8))


if __name__ == "__main__":
    unittest.main(verbosity=2)