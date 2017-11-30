# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017 Met Office.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
A plugin for creating tables from spotdata forecasts for Database export.

"""

import itertools
import pandas as pd
from pandas import DataFrame
import numpy as np
import sqlite3
import os
from datetime import datetime as dt


class SpotDatabase(object):
    """
    Holds the Spotdata Database table configuration and mapping from Cubes.

    """

    def __init__(self, output, outfile, tablename, primary_dim,
                 coord_to_slice_over,
                 primary_map=None,
                 primary_func=None,
                 pivot_dim=None,
                 pivot_map=None,
                 column_dims=None,
                 column_maps=None):
        """
        Args:
            output (str):
                Some filetype to output, currently 'sqlite' or 'csv' are valid.
            outfile (str):
                The path to the file to be output to.
            tablename (str):
                Name of the SQL table to create, if required.
            primary_dim (str):
                The dimension to use as the first index in the table.
            coord_to_slice_over (str):
                Some coordinate over which to take slices over the cubes.

        Keyword Args:
            primary_map (None or list):
                The column names to use in place of the primary dimension.
            primary_func (None or list):
                Any fucntions to apply to the primary_dim to produce the values
                in each of the primary_map columns.
            pivot_dim (None or str):
                A dimension to pivot into multiple columns
            pivot_map (None or function):
                A function to map to the values of the columns, which become
                the column names.
            column_dims (None or list):
                Any further dimensions or to be mapped from the cube.
            column_maps (None or list):
                A new name for each column_dim in the table to be output.

        """

        self.primary_dim = primary_dim
        self.primary_map = primary_map
        self.primary_func = primary_func
        self.pivot_dim = pivot_dim
        self.pivot_map = pivot_map

        self.column_dims = column_dims
        self.column_maps = column_maps
        self.output = output
        self.outfile = outfile
        self.tablename = tablename
        self.coord_to_slice_over = coord_to_slice_over

    def __repr__(self):
        """
        Representation of the pre-processed configuration of the SpotDatabase.

        """
        result = '<SpotDatabase: {output}, {outfile}, {tablename}, '\
                 '{primary_dim}, {coord_to_slice_over}, '\
                 'primary_map={primary_map}, '\
                 'primary_func={primary_func}, '\
                 'pivot_dim={pivot_dim}, '\
                 'pivot_map={pivot_map}, '\
                 'column_dims={column_dims}, '\
                 'column_maps={column_maps}>'
        return result.format(**self.__dict__)

    def to_dataframe(self, cubelist):
        """
        Turns the cubelist into a Pandas DatafFame.

        The cube is sliced, over coord_to_slice_over, with primary_dim as an
        initial index.

        If pivot dim is provided, a column is added with pivot column names,
        then the table is rotated (or pivoted) to create these columns.

        If a primary map is provided, the index is mapped to these columns,
        transforming the data using primary func.

        If column dims are provided, each are added as columns to the
        DataFrame, using the column map to determine the column names.

        Args:
            cubelist (iris.cube.CubeList):
                A Cubelist to populate the table.

        """

        for cube in cubelist:
            for cube_slice in cube.slices_over(self.coord_to_slice_over):
                self.check_input_dimensions(cube_slice)
                df = DataFrame(cube_slice.data,
                               index=cube_slice.coord(self.primary_dim).points,
                               columns=['values'])
                if self.pivot_dim:
                    # Reshape data based on column values
                    df = self.pivot_table(cube_slice, df)

                if self.primary_map:
                    self.map_primary_index(df)

                if self.column_dims and self.column_maps:
                    for dim, col in itertools.izip_longest(self.column_dims,
                                                           self.column_maps):
                        self.insert_extra_mapped_column(df, cube_slice, dim,
                                                        col)
                try:
                    self.df = self.df.combine_first(df)
                except AttributeError:
                    self.df = df

    def check_input_dimensions(self, cube):
        """
        Check that the input cube has the correct dimsions after being sliced
        along the coord_to_slice_over. In the input cube only the dimension
        we are slicing over and the pivot dimension can have multiple points
        in.

        Args:
            cube (iris.cube.Cube):
                The cube to check.
        Raises:
            ValueError: If the cube has the wrong dimensions and cannot be
                convertered to a table using this function.

        """
        shape = cube.shape
        pivot_axis = None
        if self.pivot_dim:
            pivot_axis = cube.coord_dims(self.pivot_dim)[0]

        for index, dim_length in enumerate(shape):
            if pivot_axis is not None and index == pivot_axis:
                continue
            elif dim_length is not 1:
                message = ("Dimensions that are not described by the pivot_dim"
                           " or coord_to_slice_over must only have one point "
                           "in. Dimension '{}' has length '{}' and "
                           "is associated with the '{}' coordinate.")
                message = message.format(
                    index, dim_length,
                    cube.coord(dimensions=index, dim_coords=True).name())
                raise ValueError(message)

    def pivot_table(self, cube, dataframe):
        """
        Produces a 'pivot' table by inserting the coords of the pivot dimension
        and pivoting on that column, to produce columns of names mapped
        from the cube's dimension coords.

        Args:
            cube (iris.cube.Cube):
                The cube to used to determine the coords.
            dataframe (pandas.DataFrame):
                The dataframe to modify.
        Returns:
            dataframe (pandas.DataFrame):
                The modified dataframe.

        """
        coords = cube.coord(self.pivot_dim).points
        col_names = map(self.pivot_map, coords)
        dataframe.insert(1, self.pivot_dim, col_names)
        dataframe = dataframe.pivot(columns=self.pivot_dim, values='values')
        return dataframe

    def map_primary_index(self, dataframe):
        """
        Insert into the DataFrame columns mapped from the primary index.

        Args:
            dataframe (pandas.DataFrame):
                The dataframe to modify.

        """
        for mapping, function in zip(self.primary_map,
                                     self.primary_func):
            dataframe.insert(0, mapping, map(function, dataframe.index))
        dataframe.set_index(self.primary_map, inplace=True)

    @staticmethod
    def insert_extra_mapped_column(df, cube, dim, col):
        """
        Insert into the DataFrame an extra column mapped from the cube.

        Args:
            df (pandas.DataFrame):
                The DataFrame to modify.
            cube (iris.cube.Cube):
                The cube to used to determine the coords.
            dim (str):
                The name of the dimension, attribute or constant to use for
                values of the column.
            col (str):
                The name of the column to insert.

        """
        if dim in df.columns:
            return
        # Check if dim is a coordinate on the cube, and use this coordinate for
        # the new column if available.
        elif dim in [coord.name() for coord in cube.coords()]:
            coord = cube.coord(dim)
            column_name = col
            if len(coord.points) == 1:
                column_data = coord.points[0]
            else:
                column_data = coord.points
        # Check to see if provided dim is a method or attribute of the cube.
        # Attributes are converted to a string.
        elif hasattr(cube, dim):
            attr = getattr(cube, dim)
            column_name = col
            if callable(attr):
                column_data = attr()
            else:
                column_data = str(attr)
        else:
            column_name = col
            column_data = dim
        df.insert(1, column_name, column_data)
        df.set_index(column_name, append=True, inplace=True)

    def determine_schema(self, table):
        """
        Determine the schema of the SQLite database table.
        Primary keys and datatypes are determined from the indexed columns and
        the datatypes in the DataFrame.

        Args:
            table (pandas.DataFrame):
                The name of the table's schema to create.
        Returns:
            schema (str):
                The schema definition.

        """
        # Remove the current index, and use the indexed columns for for db keys
        new_df = self.df.reset_index()
        # Find the number of columns which were indexes to index primary keys
        n_keys = len(new_df.columns) - len(self.df.columns)
        schema = pd.io.sql.get_schema(new_df, table,
                                      flavor='sqlite',
                                      keys=new_df.columns[:n_keys])
        return schema

    def create_table(self, outfile, table):
        """
        Create a SQLite datafile table.

        Args:
            outfile (str):
                The path to the database file.
            table (str):
                The name of the table to create.

        """
        schema = self.determine_schema(table)
        with sqlite3.connect(outfile) as db:
            db.execute(schema)

    def to_sql(self, outfile, table):
        """
        Output the DataFrame to SQLite database file.
        If the Database does not exist, it is created, if the table exists, it
        is appended to.

        Args:
            outfile (str):
                The path to the database file.
            table (str):
                The name of the table.

        """
        if not os.path.isfile(self.outfile):
            self.create_table(self.outfile, self.tablename)

        with sqlite3.connect(outfile) as db:
            self.df.to_sql(table, con=db, if_exists='append', index=True)

    def process(self, cubelist):
        """
        Turn the cubelist into a table, creating any required output.

        Args:
            cubelist (iris.cube.CubeList):
                A Cubelist to populate the table.

        """
        self.to_dataframe(cubelist)

        if self.output not in ["sqlite", "csv"]:
            message = ("Unrecognised output type. Current options are 'sqlite'"
                       " or 'csv', '{}' given.").format(self.output)
            raise ValueError(message)

        if self.output == 'sqlite':
            self.to_sql(self.outfile, self.tablename)

        if self.output == 'csv':
            self.df.to_csv(self.outfile)


class VerificationTable(SpotDatabase):
    """
    Represents a single Verification database table.

    This class holds the configuration of a Verification table type and
    the extra functions that need to be applied to produce the table.

    """

    def __init__(self, output, outfile, tablename, experiment_id,
                 max_forecast_leadtime):
        self.output = output
        self.outfile = outfile
        self.tablename = tablename

        self.primary_dim = "time"
        self.primary_map = ['validity_date', 'validity_time']
        self.primary_func = [lambda x: dt.utcfromtimestamp(x).date(),
                             lambda x: dt.utcfromtimestamp(x).hour*100]

        self.pivot_dim = 'forecast_period'
        self.pivot_map = lambda x: 'fcr_tplus{:03d}'.format(int(x/3600))

        self.column_dims = ['wmo_site', 'name']
        self.column_maps = ['station_id', 'cf_name']
        self.coord_to_slice_over = "index"
        self.experiment_id = experiment_id

        if experiment_id:
            self.column_dims = self.column_dims + [self.experiment_id]
            self.column_maps = self.column_maps + ["exp_id"]
        self.max_forecast_leadtime = max_forecast_leadtime

    def __repr__(self):
        """
        Representation the pre-processed configuration of the VerificationTable

        """
        result = '<VerificationTable: {output}, {outfile}, {tablename}, '\
                 '{experiment_id}, {max_forecast_leadtime}>'
        return result.format(**self.__dict__)

    def ensure_all_forecast_columns(self, dataframe):
        """
        Method to ensure all forecast columns exist in the DataFrame, adding
        any that do not. This is necessary to determine the correct schema.

        Args:
            dataframe (pandas.DataFrame):
                The DataFrame to modify.

        """
        for forecast_period in range(self.max_forecast_leadtime+1):
            forecast_column_name = self.pivot_map(forecast_period)
            if forecast_column_name not in dataframe.columns:
                dataframe[forecast_column_name] = np.nan
        dataframe.sort_index(axis=1, inplace=True)

    def to_dataframe(self, cubelist):
        """
        Turn the cubelist into a verification table.

        An extension to the parent class's method (SpotData.to_dataframe)
        ensuring that all forecast ranges that may occur are present in the
        DataFrame, and that the columns are in the correct order.

        Args:
            cubelist (iris.cube.CubeList):
                A Cubelist to populate the table.

        """
        super(VerificationTable, self).to_dataframe(cubelist)
        self.ensure_all_forecast_columns(self.df)
