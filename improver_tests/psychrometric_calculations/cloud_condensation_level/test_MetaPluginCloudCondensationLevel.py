# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown copyright. The Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
from unittest.mock import patch, sentinel

from improver.psychrometric_calculations.cloud_condensation_level import (
    MetaPluginCloudCondensationLevel,
)


@patch(
    "improver.psychrometric_calculations.psychrometric_calculations.HumidityMixingRatio"
)
@patch(
    "improver.psychrometric_calculations.cloud_condensation_level.CloudCondensationLevel"
)
def test_basic(mock_CloudCondensationLevel, mock_HumidityMixingRatio):
    """Test that the underlying plugins are called in the manner we expect."""
    model_id_attr = "mosg__model_configuration"
    plugin = MetaPluginCloudCondensationLevel(model_id_attr=model_id_attr)

    # Ensure we initialise our plugins as we expect.
    mock_HumidityMixingRatio.assert_called_once_with(model_id_attr=model_id_attr)
    mock_CloudCondensationLevel.assert_called_once_with(model_id_attr=model_id_attr)

    plugin(sentinel.cube1, sentinel.cube2, sentinel.cube3)
    mock_HumidityMixingRatio().assert_called_once_with(
        sentinel.cube1, sentinel.cube2, sentinel.cube3
    )
    mock_CloudCondensationLevel().assert_called_once_with(
        mock_HumidityMixingRatio().temperature,
        mock_HumidityMixingRatio().pressure,
        mock_HumidityMixingRatio().return_value,
    )
