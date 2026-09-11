import pathlib
from collections.abc import Iterable, Sequence
from fractions import Fraction
from logging import Logger
from typing import Any

import numpy as np
import numpy.typing as npt

from common.models import MatrixWithRhs


def output_info_level_array(logger: Logger, array: Iterable[Any], name_array: str) -> None:
    logger.info("%s: %s", name_array, " ".join(map(str, array)))


def output_info_level_2d_matrix(logger: Logger, matrix: Sequence[Sequence[Any]], name_matrix: str) -> None:
    str_matrix = ""
    n = len(matrix)
    for i in range(n):
        str_matrix += "\n" + " ".join(map(str, matrix[i]))
    logger.info("%s: %s", name_matrix, str_matrix)


def output_info_level_scalar(logger: Logger, scalar: int | float | Fraction | str, name_scalar: str) -> None:
    logger.info("%s: %s", name_scalar, str(scalar))


def get_input_file_path(path: pathlib.Path) -> pathlib.Path:
    base_dir = path.resolve().parent
    return base_dir / "input_files" / f"{path.stem}.txt"


def parse_matrix_with_rhs_from_file(path: pathlib.Path, dtype: npt.DTypeLike | None = None) -> MatrixWithRhs:
    matrix, rhs = [], []
    with open(path, encoding="utf-8") as file:
        for line in file:
            split_line = line.split()
            if len(split_line) <= 1:
                raise RuntimeError(f"Failed to parse matrix with rhs from file {path}")

            matrix.append(list(map(int, split_line[:-1])))
            rhs.append(int(split_line[-1]))

    numpy_matrix, numpy_rhs = np.array(matrix, dtype=dtype), np.array(rhs, dtype=dtype)
    return MatrixWithRhs(matrix=numpy_matrix, rhs=numpy_rhs)
