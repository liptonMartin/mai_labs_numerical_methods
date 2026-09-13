import logging
import pathlib
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

import numpy as np

from common.linal_utils import calc_vector_norm, get_block_2_on_2_from_matrix, get_complex_eigenvalues
from common.models import MatrixWithPrecision, Matrix
from common.utils import (
    get_input_file_path,
    parse_matrix_with_precision_from_file,
    output_info_level_2d_matrix,
    output_info_level_array,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class RootType(StrEnum):
    REAL = "real"
    COMPLEX = "complex"


@dataclass(frozen=True)
class QRDecomposition:
    r_matrix: np.ndarray
    q_matrix: np.ndarray


@dataclass(frozen=True)
class Eigenvalues:
    eigenvalues: list[complex | float]


def qr_decomposition(matrix: Matrix) -> QRDecomposition:
    matrix, n = matrix.matrix, matrix.n

    q_matrix = np.eye(n, dtype=float)
    r_matrix = matrix.copy()

    for i in range(n - 1):
        # Берём часть i-го столбца,
        # начиная с диагонального элемента
        b = r_matrix[i:, i].copy()

        # ||b||
        norm = calc_vector_norm(b)

        # e1 размера n - i
        e_1 = np.zeros(n - i, dtype=float)
        e_1[0] = 1

        # sign(b[0]), причём sign(0) = 1
        sign = 1 if b[0] >= 0 else -1

        # v = b + sign(b1) * ||b|| * e1
        v = b + sign * norm * e_1

        v_matrix = v[:, np.newaxis]

        # H_small = I - 2vv^T / (v^Tv)
        h_small = np.eye(n - i, dtype=float) - 2 * (v_matrix @ v_matrix.T) / (v_matrix.T @ v_matrix)

        # Встраиваем H_small в H
        h_matrix = np.eye(n, dtype=float)
        h_matrix[i:, i:] = h_small

        # A_i = H_i A_(i-1)
        r_matrix = h_matrix @ r_matrix

        # Q = Q H_i
        q_matrix = q_matrix @ h_matrix

    return QRDecomposition(
        r_matrix=r_matrix,
        q_matrix=q_matrix,
    )


def is_find_all_eigenvalues(
    current_matrix: np.ndarray, prev_matrix: np.ndarray | None, precision: Fraction | float
) -> bool:
    if prev_matrix is None:
        return False

    n = len(current_matrix)
    i = 0
    while i != n:
        # case 1: блок 1 на 1 - вещественное собственное значение
        column = current_matrix[:, i]
        column_under_diagonal = column[i + 1 :]
        norm = calc_vector_norm(column_under_diagonal)

        if norm <= precision:
            i += 1
            continue

        # case 2: блок 2 на 2 - комплексный корень
        if i != n - 1 and is_stabilize_complex_block(current_matrix, prev_matrix, i, precision):
            i += 2
            continue

        return False

    return True


def is_stabilize_complex_block(
    matrix: np.ndarray, prev_matrix: np.ndarray, index: int, precision: Fraction | float
) -> bool:
    eigenvalue_1, eigenvalue_2 = get_complex_eigenvalues(matrix, index)
    prev_eigenvalue_1, prev_eigenvalue_2 = get_complex_eigenvalues(prev_matrix, index)
    if abs(eigenvalue_1 - prev_eigenvalue_1) > precision or abs(eigenvalue_2 - prev_eigenvalue_2) > precision:
        return False

    return True


def get_eigenvalues_by_qr_decomposition(matrix_with_precision: MatrixWithPrecision) -> Eigenvalues:
    matrix, precision = matrix_with_precision.matrix, matrix_with_precision.precision
    qr = qr_decomposition(Matrix(matrix))
    prev_matrix = None
    next_a_matrix = matrix
    while not is_find_all_eigenvalues(next_a_matrix, prev_matrix, precision):
        prev_matrix = next_a_matrix
        next_a_matrix = qr.r_matrix @ qr.q_matrix
        qr = qr_decomposition(Matrix(next_a_matrix))

    eigenvalues = get_eigenvalues_from_matrix(next_a_matrix, prev_matrix, precision)
    return Eigenvalues(eigenvalues=eigenvalues)


def get_eigenvalues_from_matrix(
    current_matrix: np.ndarray, prev_matrix: np.ndarray | None, precision: Fraction | float
) -> list[float]:
    if prev_matrix is None:
        raise RuntimeError("Internal error: Prev matrix is None")

    n = len(current_matrix)
    result = []
    i = 0
    while i != n:
        # case 1: блок 1 на 1 - вещественное собственное значение
        column = current_matrix[:, i]
        column_under_diagonal = column[i + 1 :]
        norm = calc_vector_norm(column_under_diagonal)

        if norm <= precision:
            result.append(current_matrix[i][i])
            i += 1
            continue

        # case 2: блок 2 на 2 - комплексный корень
        if i != n - 1 and is_stabilize_complex_block(current_matrix, prev_matrix, i, precision):
            eigenvalues = get_complex_eigenvalues(current_matrix, i)
            result.extend(eigenvalues)
            i += 2
            continue

        raise RuntimeError("Failed get eigenvalues from r matrix!")

    return result


def main() -> None:
    for root_type in RootType:
        filename = pathlib.Path(__file__)
        base_dir = filename.resolve().parent
        name = filename.stem + "_" + root_type

        input_file_path = get_input_file_path(pathlib.Path(base_dir / name))
        matrix = parse_matrix_with_precision_from_file(input_file_path, dtype=float)

        qr = qr_decomposition(matrix)
        eigenvalues = get_eigenvalues_by_qr_decomposition(matrix)
        logger.info("%s root type: ", root_type)
        output_info_level_2d_matrix(logger, qr.r_matrix, "r_matrix")
        output_info_level_2d_matrix(logger, qr.q_matrix, "q_matrix")
        output_info_level_2d_matrix(logger, qr.q_matrix @ qr.r_matrix, "QR matrix")
        output_info_level_array(logger, eigenvalues.eigenvalues, "eigenvalues")


if __name__ == "__main__":
    main()
