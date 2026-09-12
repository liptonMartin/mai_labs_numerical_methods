from dataclasses import dataclass
from fractions import Fraction

import numpy as np


@dataclass(frozen=True)
class MatrixWithRhs:
    matrix: np.ndarray
    rhs: np.ndarray

    @property
    def n(self) -> int:
        return len(self.rhs)

    @property
    def m(self) -> int:
        return len(self.matrix[0]) if len(self.matrix) > 0 else -1


@dataclass(frozen=True)
class MatrixWithRhsPrecision(MatrixWithRhs):
    precision: float | Fraction
