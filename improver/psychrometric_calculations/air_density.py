"""Plugin to calculate air density from virtual temperature."""

from typing import Union
from improver import BasePlugin
from iris.cube import Cube, CubeList
import iris
import numpy as np

from improver.constants import R_DRY_AIR


class AirDensity(BasePlugin):
    """Calculate air density from virtual temperature."""

    def _get_pressure_cube(self, cubes: CubeList) -> Cube:
        """Extract pressure cube if present."""
        for cube in cubes:
            if cube.standard_name == "air_pressure":
                return cube
        return None

    def _get_temperature_cube(self, cubes: CubeList) -> Cube:
        """Extract virtual temperature cube."""
        for cube in cubes:
            if cube.standard_name == "virtual_temperature":
                return cube
        raise ValueError("No virtual_temperature cube provided.")

    def _to_pressure_cube(self, cube: Cube) -> Cube:
        pressure_coord = cube.coord("air_pressure")


    def process(self, inputs: Union[Cube, CubeList]) -> Cube:
        """
        Calculate air density from virtual temperature.

        If an explicit pressure cube is provided the pressure is taken from here.
        Otherwise the virtual temperature cube needs to be on pressure levels.
        The cube arguments are not checked explicitly for conformant dimensions
        but must have the same shape.


        Args:
            inputs:
                Either:
                - A single Cube (virtual temperature), or
                - A CubeList containing virtual temperature and optionally pressure.

        Returns:
            Cube containing air density (kg m-3) on the same grid/levels as virtual temmperature
        """

        # --- Normalize input to CubeList ---
        if isinstance(inputs, Cube):
            cubes = CubeList([inputs])
        else:
            cubes = inputs

        # --- Get temperature cube ---
        Tv_cube = self._get_temperature_cube(cubes)

        # --- Try to get explicit pressure cube ---
        pressure_cube = self._get_pressure_cube(cubes)

        # --- Determine pressure source ---
        if pressure_cube is not None:
            # Use explicit pressure cube
            pressure = pressure_cube.copy()
            pressure.convert_units("Pa")

            pressure_data = pressure.data

        else:
            # Try to infer from temperature cube (pressure levels case)
            try:
                pressure_coord = Tv_cube.coord("air_pressure")
            except iris.exceptions.CoordinateNotFoundError:
                raise ValueError(
                    "No pressure information supplied: "
                    "provide an air_pressure cube or use a temperature cube "
                    "on pressure levels."
                )

            pressure = pressure_coord.copy()
            pressure.convert_units("Pa")

            pressure_data = iris.util.broadcast_to_shape(
                pressure.points,
                Tv_cube.shape,
                Tv_cube.coord_dims("air_pressure"),
            )

        # --- Ensure temperature is in Kelvin ---
        Tv = Tv_cube.copy()
        Tv.convert_units("K")

        # --- Compute density ---
        density_data = pressure_data / (R_DRY_AIR * Tv.data)

        # --- Create output cube ---
        density = Tv_cube.copy(data=density_data)
        density.rename("air_density")
        density.units = "kg m-3"

        return density
