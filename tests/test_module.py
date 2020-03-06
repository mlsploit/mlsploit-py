from pydantic.error_wrappers import ValidationError
import pytest

from mlsploit import Module


def test_module_build():
    display_name = 'Test Module'
    tagline = 'This is a test module!'
    doctxt = """Long documentation for this module will go here..."""

    # test build without icon URL
    m = Module.build(
        display_name=display_name,
        tagline=tagline, doctxt=doctxt)

    assert isinstance(m, Module)
    assert m.display_name == display_name
    assert m.tagline == tagline
    assert m.doctxt == doctxt
    assert m.icon_url is None

    with pytest.raises(TypeError) as excinfo:
        m.display_name = 'Another Module'
    assert '"Module" is immutable' in str(excinfo.value)

    # test build with icon URL
    icon_url = 'https://somedomain.org/icon.jpg'

    m = Module.build(
        display_name=display_name,
        tagline=tagline, doctxt=doctxt,
        icon_url=icon_url)

    assert m.icon_url == icon_url

    # test build with malformed icon URL
    icon_url_malformed = 'this-is-not-a-url'

    with pytest.raises(ValidationError) as excinfo:
        m = Module.build(
            display_name=display_name,
            tagline=tagline, doctxt=doctxt,
            icon_url=icon_url_malformed)
    assert 'invalid or missing URL scheme' in str(excinfo.value)
