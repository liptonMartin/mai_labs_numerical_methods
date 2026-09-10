import logging
import pathlib
from collections.abc import Sequence
from fractions import Fraction
from typing import Self

import numpy as np
from numpy._typing import NDArray
from pydantic import BaseModel, Field, model_validator

from common.utils import output_info_level_array, get_input_file_path


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class TridiagonalMatrix(BaseModel):
    main_diagonal: Sequence[int] = Field(..., description="главная диагональ (b в терминах учебника)")
    lower_diagonal: Sequence[int] = Field(..., description="нижняя диагональ (a в терминах учебника)")
    upper_diagonal: Sequence[int] = Field(..., description="верхняя диагональ (с в терминах учебника)")
    rhs: Sequence[int] = Field(..., description="правая часть уравнения")

    @model_validator(mode="after")
    def validate_dimension(self) -> Self:
        if len(self.main_diagonal) != len(self.lower_diagonal) != len(self.upper_diagonal) != len(self.rhs):
            raise ValueError("Invalid dimension!")

        return self


def parse_matrix_from_file(path: pathlib.Path) -> TridiagonalMatrix:
    main_diagonal, lower_diagonal, upper_diagonal, rhs = [], [], [], []

    with open(path, encoding="utf-8") as file:
        for index, line in enumerate(file, start=-1):
            split_line = line.split()

            rhs.append(split_line[-1])

            if index == -1:  # первая строка a = 0
                main_diagonal.append(split_line[0])
                upper_diagonal.append(split_line[1])
                lower_diagonal.append(0)
                continue

            if index == len(split_line) - 3:  # последняя строка, c = 0
                lower_diagonal.append(split_line[-3])
                main_diagonal.append(split_line[-2])
                upper_diagonal.append(0)
                continue

            not_zero_numbers = split_line[index : index + 3]
            lower_num, main_num, upper_num = not_zero_numbers

            lower_diagonal.append(lower_num)
            main_diagonal.append(main_num)
            upper_diagonal.append(upper_num)

    return TridiagonalMatrix(
        main_diagonal=main_diagonal, lower_diagonal=lower_diagonal, upper_diagonal=upper_diagonal, rhs=rhs
    )


def solve_equations_system(matrix: TridiagonalMatrix) -> NDArray[np.float16]:
    main_diagonal, lower_diagonal = matrix.main_diagonal, matrix.lower_diagonal
    upper_diagonal, rhs = matrix.upper_diagonal, matrix.rhs

    n = len(main_diagonal)

    # прогоночные коэффициенты
    # P_i+1 = c_i / (b_i - a_i*P_i)
    # Q_i+1 = (a_i * Q_i - d_i) / (b_i - a_i * P_i)

    p_array = [Fraction(0)]
    q_array = [Fraction(0)]

    for i in range(n):
        denominator = -main_diagonal[i] - lower_diagonal[i] * p_array[i]

        p_next = Fraction(upper_diagonal[i], denominator)
        q_next = Fraction(lower_diagonal[i] * q_array[i] - rhs[i], denominator)

        p_array.append(p_next)
        q_array.append(q_next)

    x_array = [q_array[-1]]
    for i in range(n - 1, 0, -1):
        x_next = p_array[i] * x_array[-1] + q_array[i]
        x_array.append(x_next)
    x_array.reverse()

    output_info_level_array(logger, p_array, "P")
    output_info_level_array(logger, q_array, "Q")
    output_info_level_array(logger, x_array, "X")
    return np.array(x_array, dtype=np.float16)


def main() -> None:
    """https://planetcalc.ru/1436/?matrixA=7%20-2%200%200%200%2065%0A-3%20-7%204%200%200%2023%0A0%20-2%2015%205%200%201%0A0%200%20-2%20-12%20-8%20-58%0A0%200%200%20-3%20-10%20-8%0A"""
    input_file_path = get_input_file_path(pathlib.Path(__file__))
    matrix = parse_matrix_from_file(input_file_path)
    solve_equations_system(matrix)


if __name__ == "__main__":
    main()
