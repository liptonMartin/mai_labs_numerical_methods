import logging
import pathlib
from fractions import Fraction

import numpy as np

from common.models import MatrixWithRhs
from common.utils import get_input_file_path, parse_matrix_with_rhs_from_file, output_info_level_array

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def solve_by_fixed_point_iteration(matrix_with_rhs: MatrixWithRhs, precision: Fraction | float) -> np.ndarray:
    matrix, rhs = matrix_with_rhs.matrix, matrix_with_rhs.rhs
    n, m = matrix_with_rhs.n, matrix_with_rhs.m
    x_array = np.zeros(n, dtype=Fraction)
    next_x_array = np.zeros(n, dtype=Fraction)

    while True:
        for i, row in enumerate(matrix):
            s = sum(x_array[j] * row[j] for j in range(m) if j != i)
            next_x_array[i] = Fraction(rhs[i] - s, row[i])

        x_array = next_x_array.copy()
        d_array = next_x_array - x_array
        for d in d_array:
            if d > precision:
                continue
        break

    return next_x_array

def main() -> None:
    input_file_path = get_input_file_path(pathlib.Path(__file__))
    matrix = parse_matrix_with_rhs_from_file(input_file_path, dtype=Fraction)
    x_array_fixed_point_iteration = solve_by_fixed_point_iteration(matrix, Fraction(1, 10000000000000000000000000000000000000))
    output_info_level_array(logger, x_array_fixed_point_iteration, "x_array_fixed_point_iteration")


if __name__ == "__main__":
    main()

