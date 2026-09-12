import logging
import pathlib
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from common.models import MatrixWithPrecision
from common.utils import get_input_file_path, parse_matrix_with_precision_from_file, output_info_level_2d_matrix, \
    output_info_level_array, output_info_level_scalar

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class MaxElement:
    value: Fraction | float | int
    abs_value: Fraction | float | int
    row: int
    col: int


@dataclass(frozen=True)
class Result:
    eigenvectors_matrix: np.ndarray
    eigenvalues_matrix: np.ndarray
    eigenvalues: np.ndarray


def solve(matrix: MatrixWithPrecision) -> Result:
    n, m = matrix.n, matrix.m
    matrix, precision = matrix.matrix.copy(), matrix.precision
    eigenvectors_matrix = np.eye(n, m, dtype=Fraction)

    while (coordinates_max_element := find_max_abs_not_diagonal_element(matrix)).abs_value >= precision:
        p, q = coordinates_max_element.row, coordinates_max_element.col
        # tan (2phi) = - 2a[p][q] / (a[p][p] - a[q][q])
        phi = -0.5 * np.atan2(2 * matrix[p][q], matrix[p][p] - matrix[q][q])
        c = np.cos(phi)
        s = np.sin(phi)

        matrix_p_p, matrix_q_q, matrix_p_q = matrix[p][p], matrix[q][q], matrix[p][q]

        # except A = J^T * A * J
        matrix[p][p] = c**2 * matrix_p_p - 2 * s * c * matrix_p_q + s**2 * matrix_q_q
        matrix[q][q] = s**2 * matrix_p_p + 2 * s * c * matrix_p_q + c**2 * matrix_q_q
        matrix[p][q], matrix[q][p] = 0, 0

        for k in range(n):
            if k == p or k == q:
                continue

            matrix_p_k = matrix[p][k]
            matrix_q_k = matrix[q][k]

            matrix[p][k] = c * matrix_p_k - s * matrix_q_k
            matrix[k][p] = matrix[p][k]

            matrix[q][k] = s * matrix_p_k + c * matrix_q_k
            matrix[k][q] = matrix[q][k]

        jacobi_matrix = np.eye(n, m, dtype=Fraction)
        jacobi_matrix[p][p], jacobi_matrix[q][q] = c, c
        jacobi_matrix[p][q], jacobi_matrix[q][p] = s, -s

        eigenvectors_matrix = eigenvectors_matrix @ jacobi_matrix

    eigenvalues = np.diag(matrix)
    matrix = np.diag(eigenvalues)
    return Result(eigenvalues_matrix=matrix, eigenvectors_matrix=eigenvectors_matrix, eigenvalues=eigenvalues)


def find_max_abs_not_diagonal_element(matrix: np.ndarray) ->  MaxElement:
    max_element = 0
    row, col = -1, -1
    n = len(matrix)
    m = len(matrix[0]) if n >= 0 else -1
    for i in range(n):
        for j in range(m):
            if i == j:
                continue

            if abs(matrix[i][j]) > abs(max_element):
                row = i
                col = j
                max_element = matrix[i][j]
    return MaxElement(value=max_element, abs_value=abs(max_element), row=row, col=col)


def main() -> None:
    input_file_path = get_input_file_path(pathlib.Path(__file__))
    matrix = parse_matrix_with_precision_from_file(input_file_path, dtype=Fraction)

    result = solve(matrix)
    output_info_level_scalar(logger, matrix.precision, "precision")

    output_info_level_array(logger, result.eigenvalues, "eigenvalues")
    output_info_level_2d_matrix(logger, result.eigenvectors_matrix, "eigenvectors")

    output_info_level_2d_matrix(logger, matrix.matrix @ result.eigenvectors_matrix, "matrix * eigenvectors")
    output_info_level_2d_matrix(logger, result.eigenvectors_matrix @ result.eigenvalues_matrix, "eigenvectors * eigenvalues")

if __name__ == "__main__":
    main()
