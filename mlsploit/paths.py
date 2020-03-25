# pylint: disable=too-few-public-methods

from os import PathLike

from pathlib import Path, PurePosixPath
from typing import Union


__all__ = ["FilepathType", "JobPaths", "LibraryPaths", "ModulePaths"]


FilepathType = Union[str, PathLike]


class LibraryPaths:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.auxiliary_dir = self.base_dir / "_auxiliary"
        self.dummy_module_dir = self.auxiliary_dir / "dummy_module"
        self.module_schema_file = self.auxiliary_dir / "mlsploit_module.schema"


class ModulePaths:
    _module_dir = None

    def __init__(self):
        module_dir = self._module_dir or Path.cwd().resolve()

        self.module_dir = Path(module_dir).resolve()
        self.module_file = self.module_dir / "mlsploit_module.yaml"

    @classmethod
    def set_module_dir(cls, module_dir_path: FilepathType):
        cls._module_dir = Path(module_dir_path)

    @classmethod
    def reset_module_dir(cls):
        cls._module_dir = None


class JobPaths:
    _job_dir = None

    def __init__(self):
        job_dir = self._job_dir or PurePosixPath("/mnt")

        self.job_dir = Path(job_dir).resolve()
        self.input_dir = self.job_dir / "input"
        self.output_dir = self.job_dir / "output"
        self.input_data_file = self.input_dir / "input.json"
        self.output_data_file = self.output_dir / "output.json"

    @classmethod
    def set_job_dir(cls, job_dir_path: FilepathType):
        cls._job_dir = Path(job_dir_path)

    @classmethod
    def set_cwd_as_job_dir(cls):
        cls.set_job_dir(Path.cwd().resolve())

    @classmethod
    def reset_job_dir(cls):
        cls._job_dir = None
