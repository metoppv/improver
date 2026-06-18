"""Plugin to calculate air density from virtual temperature."""

from improver import BasePlugin
from iris.cube import Cube
import iris
import numpy as np

from improver.constants import R_DRY_AIR


class AirDensity(BasePlugin):
    """Calculate air density from virtual temperature."""

    def process(self, virtual_temperature: Cube) -> Cube:
        """
        Calculate air density from virtual temperature.

        Args:
            virtual_temperature:
                Cube of virtual temperature (K) with a pressure coordinate.

        Returns:
            Cube containing air density (kg m-3) on the same grid/levels.
        """

        # --- Extract pressure coordinate ---
        try:
            pressure_coord = virtual_temperature.coord("air_pressure")
        except iris.exceptions.CoordinateNotFoundError:
            raise ValueError(
                "Virtual temperature cube must have an 'air_pressure' coordinate."
            )

        # Ensure pressure is in Pa
        pressure = pressure_coord.copy()
        pressure.convert_units("Pa")

        # Broadcast pressure to data shape
        # (assumes pressure is a dimension coordinate)
        pressure_data = iris.util.broadcast_to_shape(
            pressure.points,
            virtual_temperature.shape,
            virtual_temperature.coord_dims("air_pressure"),
        )

        # --- Ensure temperature units are K ---
        Tv = virtual_temperature.copy()
        Tv.convert_units("K")

        # --- Compute density ---
        density_data = pressure_data / (R_DRY_AIR * Tv.data)

        # --- Create output cube ---
        density = virtual_temperature.copy(data=density_data)

        density.rename("air_density")
        density.units = "kg m-3"

        return density
