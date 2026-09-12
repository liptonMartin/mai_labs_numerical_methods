import logging
import pathlib
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

import numpy as np
from pydantic import BaseModel

from common.models import MatrixWithRhs
from common.utils import get_input_file_path, parse_matrix_with_rhs_from_file, output_info_level_array, \
    output_info_level_array_with_float_type, output_info_level_scalar

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class Method(StrEnum):
    FIXED_POINT_ITERATION = "fixed_point_iteration"
    SEIDEL_METHOD = "seidel_method"


@dataclass(frozen=True)
class Result:
    array: np.ndarray
    count_iterations: int



def solve_by_fixed_point_iteration(matrix_with_rhs: MatrixWithRhs, precision: Fraction | float, method: Method) -> Result:
    matrix, rhs = matrix_with_rhs.matrix, matrix_with_rhs.rhs
    n, m = matrix_with_rhs.n, matrix_with_rhs.m
    x_array = np.zeros(n, dtype=Fraction)
    next_x_array = np.zeros(n, dtype=Fraction)

    count_iterations = 0
    while True:
        for i, row in enumerate(matrix):
            s = Fraction(rhs[i])
            for j in range(m):
                if i == j:
                    continue
                match method:
                    case Method.FIXED_POINT_ITERATION:
                        s -= x_array[j] * row[j]
                    case Method.SEIDEL_METHOD:
                        s -= next_x_array[j] * row[j]
                    case _:
                        raise RuntimeError("Unknown method!")
            next_x_array[i] = Fraction(s, row[i])
        count_iterations += 1

        d_array = next_x_array - x_array
        x_array = next_x_array.copy()

        for d in d_array:
            if abs(d) > precision:
                break
        else:
            return Result(array=x_array, count_iterations=count_iterations)

def main() -> None:
    input_file_path = get_input_file_path(pathlib.Path(__file__))
    matrix = parse_matrix_with_rhs_from_file(input_file_path, dtype=Fraction)
    result_fixed_point_iteration = solve_by_fixed_point_iteration(matrix, Fraction(1, 1000000000000000000000000), Method.FIXED_POINT_ITERATION)
    result_seidel_method = solve_by_fixed_point_iteration(matrix, Fraction(1, 1000000000000000000000000), Method.SEIDEL_METHOD)
    output_info_level_array_with_float_type(logger, result_fixed_point_iteration.array, "x_array_fixed_point_iteration")
    output_info_level_scalar(logger, result_fixed_point_iteration.count_iterations, "count iteration fixed point iteration")
    output_info_level_array_with_float_type(logger, result_seidel_method.array, "x_array_seidel_method")
    output_info_level_scalar(logger, result_seidel_method.count_iterations, "count Seidel method iteration")


if __name__ == "__main__":
    main()

