# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
""" Provides ability to extract values from table."""


from typing import Union

import iris
import numpy as np
from pandas import DataFrame

from improver import BasePlugin
from improver.utilities.common_input_handle import as_cubelist
from improver.utilities.cube_manipulation import enforce_coordinate_ordering


class ExtractValueFromTable(BasePlugin):
    """ Plugin to extract values from a DataFrame using the provided inputs cubes to select which rows
    and columns to extract.

    The table is expected to contain numerical labels for every row and column. These labels will be
    sorted into ascending order.

    The input cubes are used to select the row and column labels from the table
    with one cube used to select the rows and the other cube used to select the columns. The values
    in the table are then extracted based on the row and column labels selected from the cubes such
    that the nearest lower label from the table is used.

    For Example:
    If the table has the following row labels of [0,1,2,3] and the row cube had a values of 1.9 then
    the row with label 1 would be selected. If the row cube had a value of 2 then the row with label
    2 would be selected. If the row cube had a value of -4 then the row with label 0 would be
    selected.
    """

    def __init__(self, row_name: str, new_name: str = None) -> None:
        """Initialise the plugin

            Args:
                row_name:
                    Name of the cube used for indexing rows.
                new_name:
                    Optional new name for the resulting cube.
                """
        self.row_name = row_name
        self.new_name = new_name

    @staticmethod
    def nearest_lower_index(
        values: np.array, sorted_table_labels: np.array
    ) -> np.array:
        """Returns the index of the nearest lower label for every element of values.
            Args:
                values:
                    Array of values to extract from table
                table_label:
                    An array of the labels along an axis of the table.
            Returns:
                An array of indices of the nearest lower label for each value in values.
        """

        sorted_index = np.searchsorted(sorted_table_labels, values, side="right") - 1

        # If floating points are being used in the table_labels and values numpy may not
        # recognise that the values are equal. This can lead to the wrong index being selected so
        # we need to check if the values are within a tolerance of the table_labels and correct the
        # index if necessary.
        tol = 1e-5
        condition = np.abs(sorted_table_labels[:, None] - values) < tol
        true_index, index = np.where(condition)

        sorted_index[index] = true_index

        # If value is below lowest table_label then index will be -1. We need to correct this to 0.
        sorted_index = np.clip(sorted_index, 0, None)

        return sorted_index

    def extract_table_values(
        self, table: DataFrame, columns_cube: iris.cube.Cube, row_cube: iris.cube.Cube
    ) -> np.array:
        """Extract values from the table based on the provided row and column cubes.
            Args:
                table:
                    DataFrame representing the table from which values are extracted.
                columns_cube:
                    Cube used to index the columns of the table.
                row_cube:
                    Cube used to index the rows of the table.
            Returns:
                Array of values extracted from the table.
        """
        shape = columns_cube.shape

        columns_data = columns_cube.data.flatten()
        row_data = row_cube.data.flatten()
        column_index = self.nearest_lower_index(
            values=columns_data, sorted_table_labels=table.columns
        )
        row_index = self.nearest_lower_index(
            values=row_data, sorted_table_labels=table.index
        )

        result = table.to_numpy()[row_index, column_index]
        result = result.reshape(shape)
        return result

    def convert_dict_to_dataframe(self, table: dict) -> DataFrame:
        """Converts a dictionary to a pandas DataFrame"""

        table = DataFrame.from_dict(table)
        table.columns = table.columns.astype(float)
        table.index = table.index.astype(float)

        table = table.reindex(sorted(table.columns), axis=1)
        table = table.reindex(sorted(table.index), axis=0)

        return table

    def process(self, *cubes: Union[iris.cube.CubeList, iris.cube.Cube], table: dict):
        """
        Process the input cubes and extract values from a table based on the provided row and
        column indices. The row name is used to identify the cube used for indexing the rows
        of the table. The new name is an optional argument that can be used to rename the
        resulting cube.

        Args:
            cubes:
                Input cubes for indexing columns and rows of the table. Exactly 2 cubes should
                be provided one which contains the values to extract from the rows and one for
                the columns.
            table:
                A dictionary representing the table from which values are extracted. Dictionary
                should be in the form: {column_name_1:{row_name_1:value, row_name_2:value},...}

        Returns:
            Cube of the same shape and metadata as the row input cubes with values extracted
            from the table based on the row and column input cubes. The cube will be re-named
            if new_name is provided.

        Raises:
            ValueError:
                - If exactly 2 cubes are not provided.
                - If the shapes of the column and row cubes do not match.
        """
        cubes = as_cubelist(*cubes)

        if len(cubes) != 2:
            raise ValueError(
                f"""Exactly 2 cubes should be provided, one for indexing columns and one for indexing rows.
                                Provided cubes are {[cube.name() for cube in cubes]}"""
            )

        row_cube = cubes.extract_cube(self.row_name)
        cubes.remove(row_cube)
        column_cube = cubes[0]
        coord_order = [coord.name() for coord in column_cube.coords()]
        enforce_coordinate_ordering(row_cube, coord_order)

        if column_cube.shape != row_cube.shape:
            raise ValueError(
                f"""Shapes of cubes do not match. Column cube shape:
                {column_cube.shape}, row cube shape: {row_cube.shape}"""
            )

        table = self.convert_dict_to_dataframe(table)
        result = self.extract_table_values(table, column_cube, row_cube)
        if result.dtype == np.float64:
            result = result.astype(np.float32)
        result_cube = row_cube.copy(data=result)
        if self.new_name:
            result_cube.rename(self.new_name)
        return result_cube
