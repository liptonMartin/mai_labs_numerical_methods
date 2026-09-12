from dataclasses import dataclass
from fractions import Fraction

import numpy as np


@dataclass(frozen=True)
class Matrix:
    matrix: np.ndarray

    @property
    def n(self) -> int:
        return len(self.matrix)

    @property
    def m(self) -> int:
        return len(self.matrix[0]) if len(self.matrix) > 0 else -1


@dataclass(frozen=True)
class MatrixWithRhs(Matrix):
    rhs: np.ndarray


@dataclass(frozen=True)
class MatrixWithPrecision(Matrix):
    precision: float | Fraction


@dataclass(frozen=True)
class MatrixWithRhsPrecision(MatrixWithRhs, MatrixWithPrecision): ...
