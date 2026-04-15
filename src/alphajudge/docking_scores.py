from __future__ import annotations

import math
from dataclasses import dataclass


def _sigmoid(x: float, L: float, x0: float, k: float, b: float) -> float:
    return L / (1 + math.exp(-k * (x - x0))) + b


@dataclass(frozen=True)
class DockQParams:
    L: float
    X0: float
    K: float
    B: float

    def score(self, x: float) -> float:
        return _sigmoid(x, self.L, self.X0, self.K, self.B)


PDOCKQ = DockQParams(0.724, 152.611, 0.052, 0.018)
PDOCKQ2 = DockQParams(1.31, 84.733, 0.075, 0.005)
MPDOCKQ = DockQParams(0.728, 309.375, 0.098, 0.262)
D0 = 10.0
