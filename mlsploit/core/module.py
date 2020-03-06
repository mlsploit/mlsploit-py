from typing import Any, List, Optional

from pydantic import HttpUrl, validator
from typing_extensions import Literal
import yamale
import yaml

from .base import FauxImmutableModel
from ..paths import FilepathType, LibraryPaths, ModulePaths


__all__ = ['Function', 'Module']


class Tag(FauxImmutableModel):
    # pylint: disable=too-few-public-methods
    name: str
    type: Literal['str', 'int', 'float']


class Option(FauxImmutableModel):
    # pylint: disable=too-few-public-methods
    name: str
    type: Literal['str', 'int', 'float', 'bool', 'enum']
    doctxt: str
    required: bool
    enum_values: Optional[List[Any]]
    default: Optional[Any]

    # pylint: disable=no-self-argument,no-self-use
    @validator('name', always=True)
    def _ensure_name_is_valid_identifier(cls, v):
        if not v.isidentifier():
            raise ValueError('option name has to be a valid python identifier')
        return v

    @validator('enum_values', always=True)
    def _check_enum_values_supplied_if_option_is_enum(cls, v, values):
        given_type = values.get('type')
        if given_type == 'enum' and (not bool(v)):
            raise ValueError('enum_values required for enum option')
        return v

    @validator('default', always=True)
    def _check_default_is_supplied_if_option_not_required(cls, v, values):
        is_required = bool(values.get('required'))
        if (not is_required) and v is None:
            raise ValueError('default cannot be empty '
                             'when required is False')
        return v

    @validator('default')
    def _cast_default_to_correct_type(cls, v, values):
        given_type = values.get('type')
        if given_type != 'enum' and v is not None:
            v = {'str': str, 'int': int,
                 'float': float, 'bool': bool
                 }[given_type](v)
        return v

    @validator('default')
    def _check_default_is_one_of_enum_values(cls, v, values):
        enum_values = values.get('enum_values')
        if enum_values is not None and v not in enum_values:
            raise ValueError(f'permitted values for default: {enum_values} '
                             f'[given: {v}]')
        return v
    # pylint: enable=no-self-argument,no-self-use


class Function(FauxImmutableModel):
    name: str
    doctxt: str
    options: List[Option] = list()
    creates_new_files: bool
    modifies_input_files: bool
    expected_filetype: str
    optional_filetypes: Optional[List[str]]
    output_tags: Optional[List[Tag]]

    @validator('options', 'output_tags', pre=True, always=True)
    def _init_list_if_not_supplied(cls, v):
        # pylint: disable=no-self-argument,no-self-use
        return v or list()

    def add_option(self,
                   name: str,
                   type: str,
                   doctxt: str,
                   required: bool,
                   enum_values: Optional[List[Any]] = None,
                   default: Optional[Any] = None
                   ) -> Option:
        # pylint: disable=redefined-builtin,too-many-arguments

        if name in {o.name for o in self.options}:
            raise RuntimeError(f'option {name} already exists '
                               f'for function {self.name}')

        o = Option(name=name,
                   type=type,
                   doctxt=doctxt,
                   required=required,
                   enum_values=enum_values,
                   default=default)

        self.options.append(o)
        return o

    def get_option(self, option_name: str) -> Option:
        filtered = [o for o in self.options
                    if o.name == option_name]

        if len(filtered) == 0:
            raise ValueError(f'cannot find option "{option_name}"')

        return filtered.pop()

    def add_output_tag(self, name: str, type: str):
        # pylint: disable=redefined-builtin
        t = Tag(name=name, type=type)
        self.output_tags.append(t)


class Module(FauxImmutableModel):
    display_name: str
    tagline: str
    doctxt: str
    functions: List[Function] = list()
    icon_url: Optional[HttpUrl]

    @validator('functions', pre=True, always=True)
    def _init_list_if_not_supplied(cls, v):
        # pylint: disable=no-self-argument,no-self-use
        return v or list()

    def add_function(self,
                     name: str,
                     doctxt: str,
                     creates_new_files: bool,
                     modifies_input_files: bool,
                     expected_filetype: str,
                     optional_filetypes: Optional[List[str]] = None
                     ) -> Function:
        # pylint: disable=too-many-arguments

        if name in {f.name for f in self.functions}:
            raise RuntimeError(f'function {name} already exists')

        f = Function(name=name, doctxt=doctxt,
                     creates_new_files=creates_new_files,
                     modifies_input_files=modifies_input_files,
                     expected_filetype=expected_filetype,
                     optional_filetypes=optional_filetypes)

        self.functions.append(f)
        return f

    def get_function(self, function_name: str) -> Function:
        filtered = [f for f in self.functions
                    if f.name == function_name]

        if len(filtered) == 0:
            raise ValueError(f'cannot find function "{function_name}"')

        return filtered.pop()

    def validate_with_schema(self, schema_path: Optional[FilepathType] = None):
        schema_path = schema_path or LibraryPaths().module_schema_file
        schema = yamale.make_schema(schema_path)
        data = [(self.dict(), self.__class__.__name__)]
        yamale.validate(schema, data, strict=True)

    def save(self, path: Optional[FilepathType] = None):
        self.validate_with_schema()

        path = path or ModulePaths().module_file
        with open(path, 'w') as f:
            f.write(yaml.dump(
                self.dict(),
                sort_keys=False,
                default_flow_style=False))

    @classmethod
    def load(cls, path: Optional[FilepathType] = None) -> 'Module':
        path = path or ModulePaths().module_file
        with open(path, 'r') as f:
            data = yaml.safe_load(f.read())

        module = cls.parse_obj(data)
        module.validate_with_schema()
        return module

    @classmethod
    def build(cls, display_name: str,
              tagline: str, doctxt: str,
              icon_url: Optional[str] = None) -> 'Module':

        data = {'display_name': display_name,
                'tagline': tagline,
                'doctxt': doctxt,
                'functions': list(),
                'icon_url': icon_url}

        return cls(**data)
