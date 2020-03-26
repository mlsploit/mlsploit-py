from devtools import debug
from pydantic.error_wrappers import ValidationError
import pytest

from mlsploit import Module
from mlsploit.core.module import Function
from mlsploit.paths import ModulePaths

from .constants import *


# TODO: add tests for validation logic


def test_module_build_without_icon_url():
    m = Module.build(
        display_name=MODULE_DISPLAY_NAME, tagline=MODULE_TAGLINE, doctxt=MODULE_DOCTXT
    )

    assert isinstance(m, Module)
    assert m.display_name == MODULE_DISPLAY_NAME
    assert m.tagline == MODULE_TAGLINE
    assert m.doctxt == MODULE_DOCTXT
    assert m.icon_url is None
    assert type(m.functions) is list and len(m.functions) == 0


@pytest.mark.parametrize("extension", ["jpg", "jpeg", "png", "svg"])
def test_module_build_with_icon_url(extension):
    m = Module.build(
        display_name=MODULE_DISPLAY_NAME,
        tagline=MODULE_TAGLINE,
        doctxt=MODULE_DOCTXT,
        icon_url=MODULE_ICON_URL.format(extension=extension),
    )

    assert m.icon_url == MODULE_ICON_URL.format(extension=extension)


def test_module_build_with_non_image_icon_url():
    icon_url_non_image = MODULE_ICON_URL.format(extension="txt")

    with pytest.raises(ValueError) as excinfo:
        Module.build(
            display_name=MODULE_DISPLAY_NAME,
            tagline=MODULE_TAGLINE,
            doctxt=MODULE_DOCTXT,
            icon_url=icon_url_non_image,
        )
    assert "not a valid image URL" in str(excinfo.value)


def test_module_build_with_malformed_icon_url():
    icon_url_malformed = "this-is-not-a-url"

    with pytest.raises(ValidationError) as excinfo:
        Module.build(
            display_name=MODULE_DISPLAY_NAME,
            tagline=MODULE_TAGLINE,
            doctxt=MODULE_DOCTXT,
            icon_url=icon_url_malformed,
        )
    assert "invalid or missing URL scheme" in str(excinfo.value)


def test_degenerate_module_is_invalid(degenerate_module):
    with pytest.raises(ValueError) as excinfo:
        degenerate_module.validate_with_schema()
    assert "functions: Length of [] is less than than 1" in str(excinfo.value)


def test_degenerate_module_cannot_be_saved(degenerate_module, tmp_module_dir):
    debug(tmp_module_dir)
    with pytest.raises(ValueError) as excinfo:
        degenerate_module.save()
    assert "functions: Length of [] is less than than 1" in str(excinfo.value)


def test_module_build_function(degenerate_module):
    m = degenerate_module

    assert type(m.functions) is list and len(m.functions) == 0

    f1 = m.build_function(
        name=FUNCTION_NAME,
        doctxt=FUNCTION_DOCTXT,
        creates_new_files=True,
        modifies_input_files=False,
        expected_filetype=FUNCTION_EXPECTED_FILETYPE,
    )

    assert type(m.functions) is list and len(m.functions) == 1
    assert m.functions[0] is f1

    f2 = m.build_function(
        name="Another Function",
        doctxt=FUNCTION_DOCTXT,
        creates_new_files=False,
        modifies_input_files=True,
        expected_filetype=FUNCTION_EXPECTED_FILETYPE,
    )

    assert len(m.functions) == 2
    assert m.functions[1] is f2
    assert f2 is not f1


def test_module_add_duplicate_function(dummy_module):
    m = dummy_module

    with pytest.raises(RuntimeError) as excinfo:
        m.build_function(
            name=FUNCTION_NAME,
            doctxt=FUNCTION_DOCTXT,
            creates_new_files=True,
            modifies_input_files=False,
            expected_filetype=FUNCTION_EXPECTED_FILETYPE,
        )
    assert "already exists" in str(excinfo)


def test_module_get_function(dummy_module):
    m = dummy_module

    f = m.get_function(FUNCTION_NAME)
    assert isinstance(f, Function)
    assert f.name == FUNCTION_NAME
    assert f.doctxt == FUNCTION_DOCTXT

    with pytest.raises(RuntimeError) as excinfo:
        m.get_function("SomeOtherFunction")
    assert "cannot find function" in str(excinfo)


def test_module_save(dummy_module, tmp_module_dir):
    m = dummy_module

    assert ModulePaths().module_dir == tmp_module_dir
    assert not ModulePaths().module_file.exists()

    m.save()
    assert ModulePaths().module_file.exists()


def test_module_load(dummy_module, tmp_module_dir):
    debug(tmp_module_dir)
    dummy_module.save()

    m = Module.load()
    assert m == dummy_module
    assert m is not dummy_module


def test_module_is_immutable(dummy_module):
    with pytest.raises(TypeError) as excinfo:
        dummy_module.display_name = "Another Module"
    assert '"Module" is immutable' in str(excinfo.value)
