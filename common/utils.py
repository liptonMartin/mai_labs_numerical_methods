import pathlib
from collections.abc import Iterable
from logging import Logger
from typing import Any


def output_info_level_array(logger: Logger, array: Iterable[Any], name_array: str) -> None:
    logger.info("%s: %s", name_array, " ".join(map(str, array)))


def get_input_file_path(path: pathlib.Path) -> pathlib.Path:
    base_dir = path.resolve().parent
    return base_dir / "input_files" / f"{path.stem}.txt"
