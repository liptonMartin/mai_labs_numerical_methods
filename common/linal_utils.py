import cmath
from collections.abc import Iterable, Sequence
from fractions import Fraction

from common.models import Block2On2


def calc_vector_norm(vector: Iterable[float | Fraction | int]) -> float:
    return sum(elem**2 for elem in vector) ** 0.5


def get_block_2_on_2_from_matrix(matrix: Sequence[Sequence[float]], index: int) -> Block2On2:
    a = matrix[index][index]
    b = matrix[index][index + 1]
    c = matrix[index + 1][index]
    d = matrix[index + 1][index + 1]
    return Block2On2(a, b, c, d)


def get_complex_eigenvalues(
    matrix: Sequence[Sequence[float]],
    index: int,
) -> tuple[complex, complex]:
    block = get_block_2_on_2_from_matrix(matrix, index)
    a, b, c, d = block.a, block.b, block.c, block.d

    trace = a + d
    determinant = a * d - b * c

    discriminant = trace**2 - 4 * determinant

    sqrt_discriminant = cmath.sqrt(discriminant)

    eigenvalue_1 = (trace + sqrt_discriminant) / 2
    eigenvalue_2 = (trace - sqrt_discriminant) / 2

    return eigenvalue_1, eigenvalue_2
