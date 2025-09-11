# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing Turbulence Index classes."""

from typing import Union, Optional
import numpy as np
import iris
iris.FUTURE.date_microseconds = True
from iris.cube import Cube, CubeList
from iris.analysis.cartography import get_xy_grids
from improver import PostProcessingPlugin
from improver.metadata.utilities import (
    create_new_diagnostic_cube,
    generate_mandatory_attributes,
)
from improver.utilities.common_input_handle import as_cubelist
from collections import namedtuple

TurbulenceInputData = namedtuple('TurbulenceInputData',
                                 ['u_wind_high_press', 'u_wind_low_press',
                                  'v_wind_high_press', 'v_wind_low_press',
                                  'geopot_high_press', 'geopot_low_press'])

class TurbulenceIndexAbove1500m_USAF(PostProcessingPlugin):
    """
    From the supplied set of cubes at two, presumable adjacent, pressure levels, calculate the
    Turbulence Index based on Ellrod 1997. This class is intended for estimates above 1500 meter.
    Turbulence Index values are typically small on the order of 1e-7 and are in units of 1/second^2 (i.e., s-2).
    """

    def _verify_compatible(self, cube1:Cube, cube2:Cube, ignore: Optional[Union[str, list[str]]]=[]) ->None:
        """
        Test a limited number of cube attributes and Cell coord attributes and verifies they are identical
        amongst the passed cubes. Raise a ValueError Exception if a test fails.

        Args:
            cube1 (iris.cube.Cube): Reference Cube
            cube2 (iris.cube.Cube): Comparison Cube
            ignore (str | [str]): An optional string or list of strings with the name(s) of attribute not to be used in the
                    comparisons.

        Returns:
            None

        Raises:
            ValueError:
                If one of the cubes is not spatially or temporally compatible.
        """
        min_coords_to_compare = ['latitude', 'longitude',
                              'forecast_period', 'forecast_reference_time',
                              'time']

        for coord_name in min_coords_to_compare:
            if cube1.coord(coord_name) != cube2.coord(coord_name):
                raise ValueError(f"Incompatible coordinates: {coord_name}")

        if ignore is None:
            ignore = []
        elif isinstance(ignore, str):
            ignore = [ignore]

        attrs_to_compare = ['shape','units']
        for attr in attrs_to_compare:
            if attr not in ignore:
                if getattr(cube1, attr) != getattr(cube2, attr):
                    raise ValueError(f"Incompatible attributes: {attr}")

    @staticmethod
    def _convert_cubelist_units(cubelist:CubeList, target_units:str) -> None:
          """
          Converts the units on the data field of all Cubes in a CubeList to the passed unit type.

          Args:
              cubelist (iris.cube.CubeList): The input CubeList.
              target_units (str): The desired units for the cubes.
          """
          for cube in cubelist:
              cube.convert_units(target_units)

    def _get_inputs(self, cubes: CubeList) -> TurbulenceInputData:
        """
        Perform validation, unit conversion, and reduction of passed Cubes.

        Args:
            cubes (iris.cube.CubeList): The passed CubeList.

        Returns:
            TurbulenceInputData object

        Raises:
            ValueError: If insufficient or compatible cubes are passed.
        """

        if len(cubes) < 6:
            raise ValueError(f"Six cubes of data: U wind, V wind, and geo potential heights; at two pressure levels"
                             f" are needed. "
                             f"{len(cubes)} cubes provided.")

        # Coerce all pressures level references to the same units.
        # Set to mb as that is used in the product name produced.
        for c in cubes:
            c.coord("pressure").convert_units('millibar')

        # Grab data and specify consistent units for comparisons and later maths.
        # Grab U components
        u_wind_constraint = iris.Constraint(cube_func=lambda cube: cube.name().startswith("UWindComponent"))
        u_winds = cubes.extract(u_wind_constraint)
        self._convert_cubelist_units(u_winds, 'm s-1')

        v_wind_constraint = iris.Constraint(cube_func=lambda cube: cube.name().startswith("VWindComponent"))
        v_winds = cubes.extract(v_wind_constraint)
        self._convert_cubelist_units(v_winds, 'm s-1')

        geopot_constraint = iris.Constraint(cube_func=lambda cube: cube.name().startswith("GeopotentialHeightAt"))
        geopots = cubes.extract(geopot_constraint)
        self._convert_cubelist_units(geopots, 'm')

        if len(u_winds) != 2:
            raise ValueError(f"Only two cubes of UWindComponents should be passed, {len(u_winds)} provided.")

        if len(v_winds) != 2:
            raise ValueError(f"Only two cubes of VWindComponents should be passed, {len(v_winds)} provided.")

        if len(geopots) != 2:
            raise ValueError(f"Only two cubes of GeopotentialHeight should be passed, {len(geopots)} provided.")

        p0_u_winds = u_winds[0].coord("pressure").cell(0).point
        p1_u_winds = u_winds[1].coord("pressure").cell(0).point
        p0_v_winds = v_winds[0].coord("pressure").cell(0).point
        p1_v_winds = v_winds[1].coord("pressure").cell(0).point
        p0_geopots = geopots[0].coord("pressure").cell(0).point
        p1_geopots = geopots[1].coord("pressure").cell(0).point

        if p0_u_winds == p1_u_winds:
            raise ValueError("Passed UWindComponents should be at two different pressure levels.")

        if p0_v_winds == p1_v_winds:
            raise ValueError("Passed VWindComponents should be at two different pressure levels.")

        if p0_geopots == p1_geopots:
            raise ValueError("Passed GeopotentialHeight should be at two different pressure levels.")

        # Test for two pressure levels and that each data type contains one at each pressure level
        p_levels = [p0_u_winds, p1_u_winds]

        if p0_v_winds not in p_levels or p1_v_winds not in p_levels:
            raise ValueError(f"Passed VWindComponents pressure levels "
                             f"{[float(p0_v_winds), float(p1_v_winds)]} inconsistent "
                             f"with UWindComponents pressure levels {[float(p_levels[0]), float(p_levels[1])]}.")

        if p0_geopots not in p_levels or p1_geopots not in p_levels:
            raise ValueError(f"Passed GeopotentialHeight pressure levels "
                             f"{[float(p0_geopots), float(p1_geopots)]} inconsistent "
                             f"with UWindComponents pressure levels {[float(p_levels[0]), float(p_levels[1])]}.")

        # Reverse list as necessary to make sure the first assignment is to the higher pressure variable
        if p0_u_winds < p1_u_winds:
            u_winds.reverse()
        u_wind_high_press = u_winds[0]
        u_wind_low_press = u_winds[1]

        if p0_v_winds < p1_v_winds:
            v_winds.reverse()
        v_wind_high_press = v_winds[0]
        v_wind_low_press = v_winds[1]

        if p0_geopots < p1_geopots:
            geopots.reverse()
        geopot_high_press = geopots[0]
        geopot_low_press = geopots[1]

        # Verify compatibility
        self._verify_compatible(u_wind_high_press, u_wind_low_press)

        self._verify_compatible(v_wind_high_press, v_wind_low_press)

        self._verify_compatible(geopot_high_press, geopot_low_press)

        self._verify_compatible(v_wind_high_press, geopot_high_press, ignore='units')

        return TurbulenceInputData(u_wind_high_press, u_wind_low_press,
                                   v_wind_high_press, v_wind_low_press,
                                   geopot_high_press, geopot_low_press)


    def process(self, cubes: Union[Cube, CubeList], model_id_attr: Optional[str] = None) -> Cube:
        """
        From the supplied set of cubes at two, presumable adjacent, pressure levels, calculate the
        Turbulence Index above 1500 m based on Ellrod 1997.
        Values are typically small on the order of 1e-7 and are in units of 1/second^2 (i.e., s-2).
        The returned Cube will have a long name beginning with "TurbulenceIndexAbove1500m" and
        concatenated with a string representing the pressure level of the calculations in millibars.
            E.g., name="TurbulenceIndexAbove1500m550mb"
        The calculations are performed on the greater pressure level (lowest altitude) provided.

        Args:
            cubes: The following six cubes are required. Additionally passed cubes are ignored.
                Cube of U component of wind at particular pressure level.
                Cube of V component of wind at identical pressure level.
                Cube of geopotential height at identical pressure level.
                Cube of U component of wind at second and adjacent pressure level.
                Cube of V component of wind at second pressure level.
                Cube of geopotential height at second pressure level.
            model_id_attr:
                Name of the attribute used to identify the source model for
                blending. This is inherited from the input temperature cube.

        Returns:
            Cube of Turbulence Index calculated at greatest provided pressure level in units of 1/second^2.

        Raises:
            ValueError:
                If one of the cubes is not spatially or temporally compatible.
        """
        cubes = as_cubelist(cubes)

        # Validate inputs and get individual elements for processing.
        (u_wind_high_press, u_wind_low_press,  # Meters per second
         v_wind_high_press, v_wind_low_press,  # Meters per second
         geopot_high_press, geopot_low_press) = self._get_inputs(cubes)  # Meters

        # Get grid point lats and longs (i.e, referenced as y and x)
        x_degs, y_degs = get_xy_grids(geopot_high_press)  # Degrees

        """
        Per USAF algorithm descriptions, Turbulence (above 1500 m)
        Taken from Ellrod and Knapp 1997: An objective clear-air turbulence forecasting technique: verification and 
        operational use. Wea. Forecasting, 7, 150-165.
        
        U wind (U) on two levels
        V wind (V) on two levels
        Vertical grid spacing in meters (Z)
        Horizontal grid spacing in meters (X,Y)
        VWS=[((dU/dZ)^2+(dV/dZ)^2)^0.5)]/d(Z)]
        DST=[dU/dX-dV/dY]
        DSH=[dV/dX-dU/dY]
        CONV=-[dU/dX+dV/dY]
        DEF=[DST^2 + DSH^2]^0.5
        Turbulence Index (TI) = VWS * (DEF + CONV)
        """

        # Presuming spherical Earth. I.e., constant WGS84 radius.
        degs_to_meters_at_eqtr = 111111  # (m/deg)

        # Get vector of all latitudes in degrees.
        lats_degs = y_degs[:,0]

        # Create a vector of scalars to address varying distances between longitudes per degree latitude.
        dx_scalar_as_func_of_latitude = np.cos(np.deg2rad(lats_degs))

        # Due to numerical inaccuracies, presuming all latitudes are in the range [-90., 90], there may be negative
        # values near +/- 90. If there are negative values, set them to zero.
        dx_scalar_as_func_of_latitude[dx_scalar_as_func_of_latitude < 0.0] = 0.0

        dx_scalar_as_func_of_latitude = dx_scalar_as_func_of_latitude.reshape(-1, 1)

        # Calculate the spatial differences across longitudes, x-axis.
        dx_deg = np.gradient(x_degs, axis=1)
        # Convert from degrees to meters
        dx_m = degs_to_meters_at_eqtr * dx_deg * dx_scalar_as_func_of_latitude

        # Calculate the spatial differences across latitudes, y-axis.
        dy_deg = np.gradient(y_degs, axis=0)
        # Convert from degrees to meters
        dy_m = degs_to_meters_at_eqtr * dy_deg

        # Calculates the vertical spatial difference between the two provided pressure levels. This is taken from the
        # geopotential height which may vary across a constant pressure level.
        # Not explicitly specified in Ellrod, but delta z in meters should be the absolute value to represent the
        # "thickness" between the two pressure levels.
        # Taking the low pressure geopotential height minus the high pressure geopotential high should ensure that.
        delta_z_m = (geopot_low_press - geopot_high_press).data

        # Deltas of wind components across a single level, for calculating deformation.
        du_mps = np.gradient(u_wind_high_press.data, axis=1)
        dv_mps = np.gradient(v_wind_high_press.data, axis=0)

        # Because of numerical issues at the poles, we may need to set certain derivative along meridians to zero.
        # Set the adjacent rows to one half of the next non zero row for smoothness.
        if abs(lats_degs[0]) == 90.0:
            dv_mps[0, :] = 0.0  # Set to zero
            dv_mps[1,:] = 0.5 * dv_mps[2,:]

        if abs(lats_degs[-1]) == 90.0:
            dv_mps[-2, :] = 0.5 * dv_mps[-3, :]
            dv_mps[-1, :] = 0.0  # Set to zero

        # DST - "stretching deformations"
        # du/dx and dv/dy are partial derivatives used to calculate "stretching deformations."
        # Units will be (1/sec).
        # I.e., the rate of change of one dimension of winds at a specific level along the same dimension.

        # Note, to prevent dividing by a spatial displacement that is unacceptably small, limit divisions to values
        # greater than one meter. If less than this value, set the partial derivative to zero.
        # This problem occurs close to the North and South poles
        du_dx = du_mps / dx_m  # Element-by-element division
        du_dx[np.abs(dx_m) < 1.0] = 0.0

        # Do the same for dy, though there is little chance that we get the near divide by zeros along the meridians.
        dv_dy = dv_mps / dy_m
        dv_dy[np.abs(dx_m) < 1.0] = 0.0
        dst = du_dx - dv_dy  # Calculate "stretching deformations."

        # DSH - "shearing deformation"
        # dvdx and dudy are partial derivatives used to calculate "shearing deformation."
        # Units will be (1/sec).
        # I.e., the rate of change of one dimension of winds at a specific level orthogonal to the specific dimension.
        du_dy = du_mps / dy_m
        du_dy[np.abs(dy_m) < 1.0] = 0.0

        dv_dx = dv_mps / dx_m
        dv_dx[np.abs(dx_m) < 1.0] = 0.0
        dsh = dv_dx + du_dy  # Calculate "shearing deformation."

        # DEF - "resultant deformation"
        # Calculate the resultant (vector RMS) deformation by combining DST and DSH (Saucier 1955).
        # Units will be (1/sec).
        # Note: deformation = (dst ** 2 + dsh ** 2) ** 0.5 will produce an object with a data type of float64.
        # To keep the passed data type (typically be float32) use Numpy functions for exponential operations.
        deformation = np.sqrt(np.square(dst)  + np.square(dsh))

        # VWS - "vertical wind shear"
        # Vertical wind shear is the vector RMS difference in winds between two layers divided
        # by the "thickness" (Ellrod 1991) between the two pressure levels. I.e., height in meters
        # Units for the deltas will be (m/sec).
        delta_u_across_layers = (u_wind_low_press - u_wind_high_press).data
        delta_v_across_layers = (v_wind_low_press - v_wind_high_press).data
        # Note: vRMS = (delta_u_across_layers**2 + delta_v_across_layers**2) ** 0.5 will produce an object with a
        # data type of float64.
        # To keep the passed data type (typically be float32) use Numpy functions for exponential operations.
        vRMS = np.sqrt(np.square(delta_u_across_layers) + np.square(delta_v_across_layers))

        vws = vRMS / delta_z_m  # Units of vws will be (1/sec).

        # CVG
        convergence = -(du_dx + dv_dy)

        # Units of turbulence_index will be (sec^-2) due to the product vws (1/sec) multiplied by
        # (deformation + convergence)(1/s)
        # As reported in Ellrod, germane index values will range form 0 to 15e-7 but higher values, e.g., 300e-7,
        # depending on the numerical model used, can be expected.
        turbulence_index = vws * (deformation + convergence)

        # Coerce to float32 which is the IMPROVER standard.
        turbulence_index = turbulence_index.astype(np.float32)

        if False:
            # For debugging:
            import matplotlib.pyplot as plt
            import matplotlib as mpl
            mpl.use('TkAgg')
            x = turbulence_index * 1e7
            plt.close()
            p = plt.imshow(x, vmax=15)
            plt.colorbar(p)
            # x = turbulence_index * 1e7 ;  plt.close(); p = plt.imshow(x, vmax=15); plt.colorbar(p)

        pressure_as_mb = u_wind_high_press.coord("pressure").cell(0).point
        name_str = f"TurbulenceIndexAbove1500mAt{int(pressure_as_mb)}mb"
        cube = create_new_diagnostic_cube(
            name=name_str,
            units="s-2",
            template_cube=u_wind_high_press,
            data=turbulence_index,
            mandatory_attributes=generate_mandatory_attributes(cubes, model_id_attr=model_id_attr)
        )

        return cube


