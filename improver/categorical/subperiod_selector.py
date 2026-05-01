# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
import iris
import numpy as np
from iris.cube import Cube

from improver import PostProcessingPlugin


class SubperiodSelector(PostProcessingPlugin):
    """
    Plugin to select which subperiods contain the phenomenon identified over the main period.

    For example, if the 50th percentile of hours of light rain over a 24 hour period is 0.25 (6 hours),
    then this plugin can be used to identify which 6 hours of the 24 hour period are most likely
    to contain light rain. The result can be used in the weather symbol decision tree to force
    the selection of a wet symbol.
    """

    def __init__(self, percentile: float, **threshold_kwargs):
        """
        Initialise the plugin.

        Args:
            percentile: The percentile of the main period diagnostic to select.
            **threshold_kwargs: Keyword arguments specifying the names and values of threshold coords
                associated with the main period diagnostic to select. One of these will also match the
                threshold coord on the subperiod diagnostic, which will be used to identify which subperiods to select.
        """
        self.percentile = percentile
        self.threshold_kwargs = threshold_kwargs

    @staticmethod
    def _pick_subperiods(
        main_period_data: np.ndarray, subperiod_data: np.ndarray
    ) -> np.ndarray:
        """Identify which subperiods to select based on the selected main period diagnostic slice.

        The value at each grid point in the main period data indicates the fraction of subperiods to select, and the values in the
        subperiod data indicate the likelihood of the phenomenon occurring in each subperiod.
        The subperiods with the highest likelihood are selected until the number of selected subperiods
        matches the value from the main period data.
        """
        number_of_subperiods = subperiod_data.shape[0]
        selected_subperiods = np.zeros_like(subperiod_data)
        for period_rank, period in enumerate(range(number_of_subperiods + 1)):
            if period == 0:
                continue
            # Identify which subperiods to select for this period rank. The leading dimension is time, so argpartition
            # is used to identify the indices of the subperiods with the highest likelihood.
            subperiods_to_select = np.argpartition(
                subperiod_data, number_of_subperiods - 1, axis=0
            )[-period_rank:]
            # Where the main_period_data indicates at least this many periods, set the selected_subperiods to 1
            selected_subperiods[subperiods_to_select] = np.where(
                main_period_data >= period / number_of_subperiods,
                1,
                selected_subperiods[subperiods_to_select],
            )
        return selected_subperiods.astype(bool)

    def _apply_constraints(
        self, main_period_cube: Cube, subperiod_cube: Cube
    ) -> (Cube, Cube):
        """Select the required cube slices"""
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
            main_period_cube: Cube containing the main period diagnostic to select.
            subperiod_cube: Cube containing the subperiod diagnostic to select.

        Returns:
            Cube indicating the selected subperiods.
        """
        main_period_slice, subperiod_slice = self._apply_constraints(
            main_period_cube, subperiod_cube
        )

        selected_periods = self._pick_subperiods(
            main_period_slice.data, subperiod_slice.data
        )
        # Create a new cube to hold the selected subperiods, using the metadata from the subperiod slice
        selected_subperiods_cube = subperiod_slice.copy(data=selected_periods)
        selected_subperiods_cube.rename("selected_subperiods")
        selected_subperiods_cube.units = "1"
        return selected_subperiods_cube
