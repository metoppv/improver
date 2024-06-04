# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
from unittest.mock import patch, sentinel

from improver.psychrometric_calculations.cloud_condensation_level import (
    MetaPluginCloudCondensationLevel,
)


@patch(
    "improver.psychrometric_calculations.cloud_condensation_level.HumidityMixingRatio"
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
