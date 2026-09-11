import logging

import numpy as np
from numpy.typing import NDArray

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def solve_by_lu_decomposition(matrix: NDArray[np.int_]) -> None:
    n = len(matrix)
    m = len(matrix[0])
    u_matrix = matrix.copy()
    l_matrix = np.eye(n, m)
    p_matrix = np.eye(n, m)
    for i in range(n):
        i_column = [abs(u_matrix[j][i]) for j in range(i, n)]
        max_abs_element = max(i_column)

        if max_abs_element == 0:
            continue

        index_max_element = i_column.index(max_abs_element) + i
        max_element = u_matrix[index_max_element][i]

        u_matrix[[i, index_max_element]] = u_matrix[[index_max_element, i]]
        p_matrix[[i, index_max_element]] = p_matrix[[index_max_element, i]]

        for j in range(i):
            l_matrix[index_max_element][j], l_matrix[i][j] = l_matrix[i][j], l_matrix[index_max_element][j]

        for j in range(i + 1, n):
            l_i_j = u_matrix[j][i] / max_element
            u_matrix[j] -= u_matrix[i] * l_i_j
            l_matrix[j][i] = l_i_j

    logger.info("U:\n%s", u_matrix)
    logger.info("L:\n%s", l_matrix)
    logger.info("P:\n%s", p_matrix)

    logger.info("P * A =\n%s", p_matrix @ matrix)
    logger.info("L * U =\n%s", l_matrix @ u_matrix)


def main() -> None:
    matrix = np.array([[2, 1, 1, 0], [4, 3, 3, 1], [8, 7, 9, 5], [6, 7, 9, 8]], dtype=np.float64)
    solve_by_lu_decomposition(matrix)


if __name__ == "__main__":
    main()