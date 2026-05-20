# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import iris
import numpy as np
from iris.cube import Cube

from improver import PostProcessingPlugin
from improver.utilities.cube_manipulation import enforce_coordinate_ordering


class SubperiodSelector(PostProcessingPlugin):
    """
    Plugin to deaggregate a fraction-of-period-that-is-xxx diagnostic into this-subperiod-is-xxx.

    For example, if light rain is expected for 6 hours within a 24 hour period (e.g. the 50th
    percentile of light rain over a 24 hour period is 0.25), this plugin selects the 6 hours
    most likely to contain that light rain. The result can be used in the weather symbol
    decision tree to force the selection of a wet symbol.
    """

    def __init__(
        self,
        percentile: float,
        new_name: str = "selected_subperiods",
        **threshold_kwargs,
    ):
        """
        Initialise the plugin.

        Args:
            percentile:
                The percentile of the main period diagnostic to select.
            new_name:
                Name of output cube.
            **threshold_kwargs:
                Keyword arguments specifying the names and values of threshold coords
                associated with the main period diagnostic to select. One of these will also
                match the threshold coord on the subperiod diagnostic, which will be used to
                identify which subperiods to select.
        """
        self.percentile = percentile
        self.threshold_kwargs = threshold_kwargs
        self.new_name = new_name

    @staticmethod
    def _pick_subperiods(
        main_period_data: np.ndarray, subperiod_data: np.ndarray
    ) -> np.ndarray:
        """Identify the subperiods most likely to contain the phenomenon.

        At each grid point, the main period data specifies the fraction of subperiods to select, and the
        subperiod data is the likelihood of the phenomenon occurring in each subperiod.
        The subperiods with the highest likelihood are selected until the number of selected subperiods
        matches the value from the main period data.

        Where multiple subperiods have equal likelihood, the selection between them is random.

        Args:
            main_period_data:
                2D array containing the main period diagnostic slice, with values indicating the fraction of
                subperiods to select.
            subperiod_data:
                3D array containing the subperiod diagnostic slice, with values indicating the likelihood of
                the phenomenon occurring in each subperiod.

        Returns:
            3D array with the same shape as subperiod_data, where the selected subperiods are indicated by 1
            and the non-selected subperiods are indicated by 0.
        """
        number_of_subperiods = subperiod_data.shape[0]
        selected_subperiods = np.zeros_like(subperiod_data)
        # The use of np.meshgrid comes from Copilot and prevents Numpy broadcasting issues when indexing
        # selected_subperiods with subperiods_to_select, which has a different shape to main_period_data.
        idx_y, idx_x = np.meshgrid(
            np.arange(subperiod_data.shape[1]),
            np.arange(subperiod_data.shape[2]),
            indexing="ij",
        )
        for period_rank, period in enumerate(range(number_of_subperiods + 1)):
            if period == 0:
                continue
            # Identify which subperiods to select for this period rank. The leading dimension is time, so argpartition
            # is used to identify the indices of the subperiods with the highest likelihood.
            subperiods_to_select = np.argpartition(
                subperiod_data, range(number_of_subperiods), axis=0
            )[-period_rank:]
            # Where the main_period_data indicates at least this many periods, set the selected_subperiods to 1
            selected_subperiods[subperiods_to_select, idx_y, idx_x] = np.where(
                main_period_data >= period / number_of_subperiods,
                1,
                selected_subperiods[subperiods_to_select, idx_y, idx_x],
            )
        return selected_subperiods.astype(np.int8)

    def _apply_constraints(
        self, main_period_cube: Cube, subperiod_cube: Cube
    ) -> tuple[Cube, Cube]:
        """
        Select the required cube slices

        The main period cube is expected to have a percentile coordinate and one or more threshold coordinates,
        and the subperiod cube is expected to have a time coordinate and one or more threshold coordinates that
        match those on the main period cube.
        The required slice is selected from the main period cube using the percentile and threshold constraints,
        and the required slice is selected from the subperiod cube using whichever of the threshold constraints
        matches a coordinate on the subperiod cube.

        Args:
            main_period_cube:
                Cube containing the main period diagnostic to select, with a percentile coordinate and one or
                more threshold coordinates.
            subperiod_cube:
                Cube containing the subperiod diagnostic to select, with a time coordinate and one or more
                threshold coordinates that match those on the main period cube.

        Returns:
            main_period_slice:
                The selected slice from the main period cube.
            subperiod_slice:
                The selected slice from the subperiod cube.

        Raises:
            ValueError:
                - If no data is found in the main period cube matching the percentile and threshold constraints.
                - If no matching threshold coordinate is found on the subperiod cube.
                - If no data is found in the subperiod cube matching the threshold constraints.
                - If the subperiod cube does not have exactly one more dimension than the main period cube.

        """
        # Select the required main period diagnostic slice
        constraints = iris.Constraint(percentile=self.percentile)
        for key, value in self.threshold_kwargs.items():
            constraints &= iris.Constraint(**{key: value})
        main_period_slice = main_period_cube.extract(constraints)
        if not main_period_slice:
            raise ValueError(
                f"No data found in main period cube matching percentile: {self.percentile} {self.threshold_kwargs}"
            )

        # Select the required subperiod diagnostic slice using whichever threshold coord exists on the subperiod cube
        subperiod_constraints = None
        for key, value in self.threshold_kwargs.items():
            if subperiod_cube.coords(key):
                subperiod_constraints = iris.Constraint(**{key: value})
                break
        if subperiod_constraints is None:
            raise ValueError(
                f"No matching threshold coordinate found on subperiod cube for keys: {list(self.threshold_kwargs.keys())}"
            )
        subperiod_slice = subperiod_cube.extract(subperiod_constraints)
        if not subperiod_slice:
            raise ValueError(
                f"No data found in subperiod cube matching threshold constraints: {subperiod_constraints}"
            )

        if not len(main_period_slice.shape) + 1 == len(subperiod_slice.shape):
            raise ValueError(
                f"Expected subperiod cube to have exactly one more dimension ('time') than main period cube, but got {main_period_slice.shape} and {subperiod_slice.shape}"
            )
        return main_period_slice, subperiod_slice

    def process(self, main_period_cube: Cube, subperiod_cube: Cube) -> Cube:
        """
        Select the subperiods most likely to contain the phenomenon identified over the main period.

        Args:
            main_period_cube:
                Cube containing the main period diagnostic to select.
            subperiod_cube:
                Cube containing the subperiod diagnostic to select.

        Returns:
            A cube of subperiods marked as 1 (is) or 0 (is not) representative of the phenomenon.
        """
        main_period_slice, subperiod_slice = self._apply_constraints(
            main_period_cube, subperiod_cube
        )
        coord_names = ["time"] + [c.name() for c in main_period_slice.dim_coords]
        enforce_coordinate_ordering(subperiod_slice, coord_names)

        selected_periods = self._pick_subperiods(
            main_period_slice.data, subperiod_slice.data
        )
        # Create a new cube to hold the selected subperiods, using the metadata from the subperiod slice
        selected_subperiods_cube = subperiod_slice.copy(data=selected_periods)
        selected_subperiods_cube.rename(self.new_name)
        selected_subperiods_cube.units = "1"
        return selected_subperiods_cube
