# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the VirtualTemperature plugin"""

from improver.temperature.virtual_temperature import (
    VirtualTemperatureFromSpecificHumidity,
)


class TestInitialisation:
    def test_init_for_virtual_temperature_specific_humidity(self):
        result = VirtualTemperatureFromSpecificHumidity()
        assert result.temperature is None
        assert result.specific_humidity is None
        assert result.cloud_water_mixing_ratio is None
        assert result.cloud_ice_mixing_ratio is None
