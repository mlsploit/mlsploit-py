from collections import namedtuple
import json
from pathlib import Path
from typing import Any, List, Mapping, MutableMapping, Optional

from pydantic import validator

from .base import FauxImmutableModel
from .module import Function, Module
from ..paths import FilepathType, JobPaths


__all__ = ["Job"]


class InputFileItem(FauxImmutableModel):
    name: str
    tags: Mapping[str, Any]

    @property
    def filetype(self) -> str:
        return Path(self.name).suffix[1:]

    @property
    def path(self) -> FilepathType:
        return JobPaths().input_dir / self.name

    def get_tag(self, tag: str) -> Any:
        return self.tags.get(tag, None)

    def has_tag(self, tag: str) -> bool:
        return self.get_tag(tag) is not None

    def validate_with_function(self, function: Function):
        # validate input file is of allowed filetype
        allowed_filetypes = {function.expected_filetype}.union(
            function.optional_filetypes or []
        )
        if self.filetype not in allowed_filetypes:
            raise TypeError(
                f"allowed filetypes: {str(allowed_filetypes)} " f"(given: {self.name})"
            )


class InputDocument(FauxImmutableModel):
    name: str
    num_files: int
    files: List[str]
    options: Mapping[str, Any]
    tags: List[Mapping[str, Any]]

    @property
    def module(self) -> Module:
        return Module.load()

    @property
    def function(self) -> Function:
        return self.module.get_function(self.name)

    @property
    def input_file_items(self) -> List[InputFileItem]:
        num_files, num_tags = len(self.files), len(self.tags)
        if num_files != num_tags:
            raise RuntimeError(f"got {num_files} files and {num_tags} tags")

        return [
            InputFileItem(name=filename, tags=tags)
            for filename, tags in zip(self.files, self.tags)
        ]

    def validate_document(self):
        function = self.function

        # check required options are present
        for option in function.options:
            if option.required and option.name not in self.options.keys():
                raise KeyError(f'required option "{option.name}" not found')

        # check each option has correct type (or value in case of enum)
        for option_name, value in self.options.items():
            option = function.get_option(option_name)

            if option.required and value is None:
                raise ValueError(
                    f'required option "{option_name}" cannot ' f"have a null value"
                )

            if option.type == "enum" and value not in option.enum_values:
                raise ValueError(
                    f'Incorrect value "{value}" '
                    f'for enum option "{option_name}". '
                    f"Allowed values: {str(option.enum_values)}"
                )

            if (
                option.type != "enum"
                and value is not None
                and type(value).__name__ != option.type
            ):
                raise ValueError(
                    f'Incorrect type of value "{value}" '
                    f'for option "{option_name}". '
                    f"Allowed type: {option.type}"
                )

        # check files
        for input_file_item in self.input_file_items:
            if not input_file_item.path.exists():
                raise RuntimeError(
                    f"cannot find input file item " f"on disk: {input_file_item.path}"
                )

            input_file_item.validate_with_function(function)

    @classmethod
    def load(cls, path: Optional[FilepathType] = None) -> "InputDocument":
        path = path or JobPaths().input_data_file
        with open(path, "r") as f:
            data = json.load(f)

        input_document = cls.parse_obj(data)
        input_document.validate_document()
        return input_document


class OutputFileItem(FauxImmutableModel):
    name: str
    is_new_file: bool
    is_modified_file: bool
    tags: MutableMapping[str, Any] = dict()

    # pylint: disable=no-self-argument,no-self-use
    @validator("tags", pre=True, always=True)
    def _init_dict_if_not_supplied(cls, v):
        return v or dict()

    @validator("is_modified_file", always=True)
    def _check_either_new_or_modified(cls, v, values):
        is_new_file = values["is_new_file"]
        if v == is_new_file:
            raise ValueError(
                "is_new_file and is_modified_file " "cannot have the same truth value"
            )
        return v

    # pylint: enable=no-self-argument,no-self-use

    @property
    def path(self) -> FilepathType:
        return JobPaths().output_dir / self.name

    def add_tag(self, name: str, value: Any):
        self.tags[name] = value

    def validate_with_function(self, function: Function):
        allowed_tags = {t.name: t.type for t in (function.output_tags or [])}

        for name, value in self.tags.items():
            if name not in allowed_tags.keys():
                raise RuntimeError(
                    f"invalid tag for " f"output file {self.name}: {name}"
                )

            if type(value).__name__ != allowed_tags[name]:
                raise ValueError(
                    f"output tag {name} should be "
                    f" {allowed_tags[name]}, "
                    f"found {type(value).__name__} "
                    f"for output file {self.name}"
                )


class OutputDocument:
    def __init__(self, function: Function):
        self._function: function = function
        self._output_file_items: List[OutputFileItem] = list()

    @property
    def name(self) -> str:
        return self._function.name

    def add_output_file_item(self, output_file_item: OutputFileItem):
        if output_file_item.name in {o.name for o in self._output_file_items}:
            raise RuntimeError(
                f"output file with same name found: " f" {output_file_item.name}"
            )

        self._output_file_items.append(output_file_item)

    def dict(self) -> dict:
        files = list()
        files_created = list()
        files_modified = list()
        tags = list()

        for output_file_item in self._output_file_items:
            filename = output_file_item.name
            files.append(filename)

            if output_file_item.is_new_file:
                files_created.append(filename)

            elif output_file_item.is_modified_file:
                files_modified.append(filename)

            else:
                raise RuntimeError(
                    f"{filename} is neither marked as " f"new nor as modified"
                )

            tags.append(output_file_item.tags or dict())

        return {
            "name": self.name,
            "files": files,
            "files_created": files_created,
            "files_modified": files_modified,
            "tags": tags,
        }

    def validate_document(self):
        all_output_file_names = set()
        for output_file_item in self._output_file_items:
            output_file_name = output_file_item.name
            output_file_path = Path(output_file_item.path)

            # check duplicates
            if output_file_name in all_output_file_names:
                raise RuntimeError(
                    f"found duplicate output file item: " f"{output_file_name}"
                )
            all_output_file_names.add(output_file_name)

            # check validity with function
            output_file_item.validate_with_function(self._function)

            # check output file item exists
            if not output_file_path.exists():
                raise RuntimeError(
                    f"cannot find output file item " f"on disk: {output_file_name}"
                )

            # check output file item is a valid file
            if not output_file_path.is_file():
                raise RuntimeError(
                    f"cannot parse output file item "
                    f"as a valid file: {output_file_name}"
                )

    def save(self, path: Optional[FilepathType] = None):
        self.validate_document()

        path = path or JobPaths().output_data_file
        with open(path, "w") as f:
            json.dump(self.dict(), f, indent=2)


class JobMeta(type):
    _input_document: InputDocument
    _output_document: OutputDocument

    @property
    def module(cls) -> Module:
        return Module.load()

    @property
    def function_name(cls) -> str:
        return cls._input_document.function.name

    @property
    def options(cls) -> namedtuple:
        options_data = cls._input_document.options
        options = namedtuple("Options", options_data.keys())
        options = options(**options_data)
        return options

    @property
    def input_file_items(cls) -> List[InputFileItem]:
        return cls._input_document.input_file_items


class Job(metaclass=JobMeta):
    _initialized: bool = False
    _committed: bool = False

    def __init__(self):
        raise NotImplementedError("use Job.initialize instead")

    @classmethod
    def initialize(cls):
        if cls._committed:
            raise RuntimeError("cannot initialize, job is already committed")

        if cls._initialized:
            return

        input_document = InputDocument.load()
        function = input_document.function
        output_document = OutputDocument(function)

        cls._input_document = input_document
        cls._output_document = output_document
        cls._initialized = True

    @classmethod
    def reserve_output_file_item(
        cls,
        output_file_name: str,
        is_new_file: bool = False,
        is_modified_file: bool = False,
        tags: Optional[Mapping[str, Any]] = None,
    ):

        if not cls._initialized:
            raise RuntimeError(
                "cannot reserve output file item " "without calling Job.initialize"
            )

        if cls._committed:
            raise RuntimeError(
                "cannot reserve output file item, " "job is already committed"
            )

        tags = tags or dict()
        output_file_item = OutputFileItem(
            name=output_file_name,
            is_new_file=is_new_file,
            is_modified_file=is_modified_file,
            tags=dict(tags),
        )

        cls._output_document.add_output_file_item(output_file_item)
        return output_file_item

    @classmethod
    def commit_output(cls):
        if not cls._initialized:
            raise RuntimeError("cannot commit job before " "calling Job.initialize")

        if cls._committed:
            return

        cls._output_document.save()
        cls._committed = True

    @classmethod
    def reset(cls):
        cls._input_document = None
        cls._output_document = None

        cls._initialized = False
        cls._committed = False
