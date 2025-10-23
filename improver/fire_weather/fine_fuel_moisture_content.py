# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import numpy as np

from improver.plugin_base import BasePlugin


class FineFuelMoistureContent(BasePlugin):
    """
    Plugin to calculate the Fine Fuel Moisture Code (FFMC) following the Canadian Forest Fire Weather Index System.
    """

    def process(
        self,
        temperature: float,
        precipitation: float,
        relative_humidity: float,
        wind_speed: float,
        input_ffmc: float,
    ) -> float:
        """
        Calculate the Fine Fuel Moisture Code (FFMC).

        Args:
            temperature (float): Air temperature in degrees Celsius.
            precipitation (float): Precipitation in mm.
            relative_humidity (float): Relative humidity in percent.
            wind_speed (float): Wind speed in km/h.
            input_ffmc (float): Previous day's FFMC value.

        Returns:
            float: Calculated FFMC value.
        """
        # Step 1: Convert previous FFMC to moisture content
        m_o = 147.2 * (101.0 - input_ffmc) / (59.5 + input_ffmc)

        # Step 2: Rainfall adjustment
        if precipitation > 0.5:
            r_f = precipitation - 0.5
            m_o = m_o + 42.5 * r_f * np.exp(-100.0 / (251.0 - m_o)) * (
                1 - np.exp(-6.93 / r_f)
            )
            if m_o > 250.0:
                m_o = 250.0

        # Step 3: Drying phase
        E_d = (
            0.942 * relative_humidity**0.679
            + 11 * np.exp((relative_humidity - 100) / 10)
            + 0.18 * (21.1 - temperature) * (1 - np.exp(-0.115 * relative_humidity))
        )

        if m_o < E_d:
            # Drying rate
            k_o = 0.424 * (1 - (relative_humidity / 100.0) ** 1.7) + 0.0694 * np.sqrt(
                wind_speed
            ) * (1 - (relative_humidity / 100.0) ** 8)
            k_d = k_o * 0.581 * np.exp(0.0365 * temperature)
            m = E_d + (m_o - E_d) * 10 ** (-k_d)
        else:
            # Wetting equilibrium
            E_w = (
                0.618 * relative_humidity**0.753
                + 10.0 * np.exp((relative_humidity - 100.0) / 10.0)
                + 0.18 * (21.1 - temperature) * (1 - np.exp(-0.115 * relative_humidity))
            )
            k_l = 0.424 * (
                1 - ((100.0 - relative_humidity) / 100.0) ** 1.7
            ) + 0.0694 * np.sqrt(wind_speed) * (
                1 - ((100.0 - relative_humidity) / 100.0) ** 8
            )
            k_w = k_l * 0.581 * np.exp(0.0365 * temperature)
            m = E_w - (E_w - m_o) * 10 ** (-k_w)

        # Step 4: Convert moisture content back to FFMC
        output_ffmc = 59.5 * (250.0 - m) / (147.2 + m)
        return output_ffmc
