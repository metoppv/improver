import pytest

from improver import api


@pytest.mark.parametrize("module", (api.PROCESSING_MODULES.keys()))
def test_all_values_are_callable(module):
    """Checks that each item in the PROCESSING_MODULES dict points to a callable object"""
    assert callable(getattr(api, module))
