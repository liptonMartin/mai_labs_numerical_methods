from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MatrixWithRhs:
    matrix: np.ndarray
    rhs: np.ndarray
