# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import pytest

from improver import api


@pytest.mark.parametrize("module", (api.PROCESSING_MODULES.keys()))
def test_all_values_are_callable(module):
    """Checks that each item in the PROCESSING_MODULES dict points to a callable object"""
    assert callable(getattr(api, module))
