#!/usr/bin/env python
# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to extract values from a table."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    *cubes: cli.inputcube, table: cli.inputjson, row_name: str, new_name: str = None,
):
    """ Extract values from a table based on the provided row and column cubes.

    The data from the row cube is compared to the labels of the rows in the table of which
    the nearest lower label is selected. The same is done for the columns using the column cube.
    The value corresponding to the selected row and column is then extracted from the table. This
    is done for every grid square.

    Args:
        cubes (iris.cube.CubeList):
            A list of iris cubes that should contain exactly two cubes: a cube from
            which the data determines which rows to extract and a cube from which the data
            determines which columns to extract.
        table(dict):
            A json file containing a dictionary representing a table of data
            with numerical row and column labels. The rows and columns should be
            in numerical order. The format of the dictionary should be:
            {column_name_1:{row_name_1:value, row_name_2:value},...}
        row_name (str):
            A string to identify the cube to be used for selecting rows.
        new_name (str):
            If provided the resulting cube will be renamed.

    Returns:
        iris.cube.Cube:
            A cube containing the extracted values from the table. The metadata will match
            the row cube but will be renamed if new_name is provided.
    """
    from improver.utilities.extract_from_table import ExtractValueFromTable

    return ExtractValueFromTable(row_name=row_name, new_name=new_name)(
        cubes, table=table
    )
