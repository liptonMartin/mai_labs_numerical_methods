import logging
import math
import pathlib
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from common.utils import (
    output_info_level_array,
    output_info_level_2d_matrix,
    output_info_level_scalar,
    get_input_file_path,
    parse_matrix_with_rhs_from_file,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LUPDecomposition:
    l_matrix: np.ndarray
    u_matrix: np.ndarray
    p_matrix: np.ndarray
    count_permutations: int


def lu_decomposition(matrix: np.ndarray) -> LUPDecomposition:
    n = len(matrix)
    m = len(matrix[0])
    u_matrix = matrix.copy()
    l_matrix = np.eye(n, m, dtype=Fraction)
    p_matrix = np.eye(n, m, dtype=Fraction)
    count_permutations = 0
    for i in range(n):
        i_column = [abs(u_matrix[j][i]) for j in range(i, n)]
        max_abs_element = max(i_column)

        if max_abs_element == 0:
            continue

        index_max_element = i_column.index(max_abs_element) + i
        max_element = u_matrix[index_max_element][i]

        u_matrix[[i, index_max_element]] = u_matrix[[index_max_element, i]]
        p_matrix[[i, index_max_element]] = p_matrix[[index_max_element, i]]
        count_permutations += index_max_element != i

        for j in range(i):
            l_matrix[index_max_element][j], l_matrix[i][j] = l_matrix[i][j], l_matrix[index_max_element][j]

        for j in range(i + 1, n):
            j_i_element = Fraction(u_matrix[j][i])
            fraction_max_element = Fraction(max_element)
            l_i_j = j_i_element / fraction_max_element
            u_matrix[j] -= u_matrix[i] * l_i_j
            l_matrix[j][i] = l_i_j

    return LUPDecomposition(
        l_matrix=l_matrix, u_matrix=u_matrix, p_matrix=p_matrix, count_permutations=count_permutations
    )


def _solve_equation_lower_matrix_on_vector(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Решение уравнения вида Lx = b, где L - нижнетреугольная матрица, где на главной диагонали единицы"""
    n = len(rhs)
    x_array = np.zeros(n, dtype=Fraction)
    for i in range(n):
        s = sum(matrix[i][j] * x_array[j] for j in range(i))
        x_array[i] = rhs[i] - s
    return x_array


def _solve_equation_upper_matrix_on_vector(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Решение уравнения вида Ux = b, где U - верхнетреугольная матрица"""
    n = len(rhs)
    x_array = np.zeros(n, dtype=Fraction)
    for i in range(n - 1, -1, -1):
        s = sum(matrix[i][j] * x_array[j] for j in range(n - 1, i - 1, -1))
        x_array[i] = Fraction(rhs[i] - s, matrix[i, i])
    return x_array


def equation_system_by_lu(decomposition: LUPDecomposition, rhs: np.ndarray) -> np.ndarray:
    """Ax = b => PAx = Pb => LUx = Pb => (Ux = y) Ly = Pb => Ly = b` => y = ...
    (возвращаемся к замене) Ux = y => x = ...
    """
    l_matrix, u_matrix, p_matrix = decomposition.l_matrix, decomposition.u_matrix, decomposition.p_matrix
    perm_rhs = p_matrix @ rhs

    y_array = _solve_equation_lower_matrix_on_vector(l_matrix, perm_rhs)
    x_array = _solve_equation_upper_matrix_on_vector(u_matrix, y_array)

    return x_array


def inverse_matrix_by_lu(decomposition: LUPDecomposition) -> np.ndarray:
    """
    A^-1 = X
    AX = E
    PAX = PE
    LUX = PE
    (Y = UX)
    LY = PE => Y = ...
    (обратная замена) UX = Y => X = ...
    """
    l_matrix, u_matrix, p_matrix = decomposition.l_matrix, decomposition.u_matrix, decomposition.p_matrix
    n = len(l_matrix)
    m = len(l_matrix[0])
    inverse_matrix = np.zeros((m, n), dtype=Fraction)
    for i in range(n):
        e_i = np.zeros(m, dtype=Fraction)
        e_i[i] = 1
        e_i = p_matrix @ e_i

        y_array = _solve_equation_lower_matrix_on_vector(l_matrix, e_i)
        x_array = _solve_equation_upper_matrix_on_vector(u_matrix, y_array)

        for j in range(m):
            inverse_matrix[j][i] = x_array[j]

    return inverse_matrix


def get_determinant_by_lu_decomposition(decomposition: LUPDecomposition) -> Fraction:
    u_matrix, count_perm = decomposition.u_matrix, decomposition.count_permutations
    n = len(u_matrix)
    return (-1) ** count_perm * math.prod(u_matrix[i][i] for i in range(n))


def solve_by_lu_decomposition(matrix: np.ndarray, rhs: np.ndarray) -> None:
    decomposition = lu_decomposition(matrix)
    l_matrix, u_matrix, p_matrix = decomposition.l_matrix, decomposition.u_matrix, decomposition.p_matrix
    solutions = equation_system_by_lu(decomposition, rhs)
    inverse_matrix = inverse_matrix_by_lu(decomposition)
    determinant = get_determinant_by_lu_decomposition(decomposition)

    output_info_level_2d_matrix(logger, u_matrix, "1.1. U")
    output_info_level_2d_matrix(logger, l_matrix, "1.2. L")
    output_info_level_2d_matrix(logger, l_matrix @ u_matrix, "2. L*U")
    output_info_level_array(logger, solutions, "3. x")
    output_info_level_2d_matrix(logger, inverse_matrix, "4. A^-1")
    output_info_level_scalar(logger, determinant, "5. det A")
    output_info_level_2d_matrix(logger, matrix @ inverse_matrix, "6.1. AA^-1")
    output_info_level_2d_matrix(logger, inverse_matrix @ matrix, "6.2. A^-1*A")
    output_info_level_2d_matrix(logger, p_matrix @ matrix, "7. P*A")


def main() -> None:
    input_file_path = get_input_file_path(pathlib.Path(__file__))
    matrix_with_rhs = parse_matrix_with_rhs_from_file(input_file_path, Fraction)
    solve_by_lu_decomposition(matrix_with_rhs.matrix, matrix_with_rhs.rhs)


if __name__ == "__main__":
    main()
