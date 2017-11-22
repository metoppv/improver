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
    Class to create a Database table from a SpotData iris.cube.

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
        Initialise class.

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
        Representation of the instance.

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

    def check_input_dimensions(self, cube):
        """
        Check that the input cube has the correct dimsions after being sliced
        along the coord_to_slice_over. In the input cube only a the dimension
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

        for i, dim_length in enumerate(shape):
            if pivot_axis is not None and i == pivot_axis:
                continue
            elif dim_length is not 1:
                message = ("Dimensions that are not described by the pivot_dim"
                           " or coord_to_slice_over must only have one point "
                           "in. Dimension '{}' has length '{}' and "
                           "is associated with the '{}' coordinate.")
                message = message.format(
                    i, dim_length,
                    cube.coord(dimensions=i, dim_coords=True).name())
                raise ValueError(message)

    def pivot_table(self, cube, df):
        """Pivots the table based on the """
        coords = cube.coord(self.pivot_dim).points
        col_names = map(self.pivot_map, coords)
        df.insert(1, self.pivot_dim, col_names)
        df = df.pivot(columns=self.pivot_dim, values='values')
        return df

    def map_primary_index(self, df):
        """Place holder docstring"""
        # Switch the index out for a map if specified
        # Has to have "time" as index if time is primary index.
        for mapping, function in zip(self.primary_map,
                                     self.primary_func):
            df.insert(0, mapping, map(function, df.index))
        # Takes significant time if a multi-index
        df.set_index(self.primary_map, inplace=True)

    @staticmethod
    def insert_extra_mapped_columns(df, cube, dim, col):
        """Place holder docstring"""
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

    def to_dataframe(self, cubelist):
        """
        Turn the input cubes into a 2-dimensional DataFrame object

        """

        for cube in cubelist:
            for c in cube.slices_over(self.coord_to_slice_over):
                self.check_input_dimensions(c)
                df = DataFrame(c.data,
                               index=c.coord(self.primary_dim).points,
                               columns=['values'])
                if self.pivot_dim:
                    # Reshape data based on column values
                    df = self.pivot_table(c, df)

                if self.primary_map:
                    self.map_primary_index(df)

                if self.column_dims and self.column_maps:
                    for dim, col in itertools.izip_longest(self.column_dims,
                                                           self.column_maps):
                        self.insert_extra_mapped_columns(df, c, dim, col)
                try:
                    self.df = self.df.combine_first(df)
                except AttributeError:
                    self.df = df

    def determine_schema(self, table):
        """
        Method to determine the schema of the database.

        Primary keys and datatypes are determined from the indexed columns and
        the datatypes in the DataFrame.

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
        Method to first create a the SQL datafile table.

        """

        schema = self.determine_schema(table)
        with sqlite3.connect(outfile) as db:
            db.execute(schema)

    def to_sql(self, outfile, table):
        """
        Output the dataframe to SQL database file.

        """
        if not os.path.isfile(self.outfile):
            self.create_table(self.outfile, self.tablename)

        with sqlite3.connect(outfile) as db:
            self.df.to_sql(table, con=db, if_exists='append', index=True)

    def process(self, cubelist):
        """
        Method to perform the table creation and output to file.

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
    Represents a single Verification database table

    """

    def __init__(self, output, outfile, tablename, experiment_ID,
                 max_forecast_lead_time):
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

        self.experiment_ID = experiment_ID

        if experiment_ID:
            self.column_dims = self.column_dims + [self.experiment_ID]
            self.column_maps = self.column_maps + ["exp_id"]
        self.max_forecast_lead_time = max_forecast_lead_time

    def __repr__(self):
        """
        Representation of the instance.

        """
        result = '<VerificationTable: {output}, {outfile}, {tablename}, '\
                 '{experiment_ID}, {max_forecast_lead_time}>'
        return result.format(**self.__dict__)

    def ensure_all_pivot_columns(self, dataframe):
        """
        Method to ensure all pivot columns exist in the dataframe, adding any
        that do not.

        """

        if self.pivot_dim and self.max_forecast_lead_time:
            for pivot_val in range(self.max_forecast_lead_time+1):
                if not self.pivot_map(pivot_val) in dataframe.columns:
                    dataframe[self.pivot_map(pivot_val)] = np.nan

    def to_dataframe(self, cubelist):
        """Add an extra method call to the to_dataframe above"""
        super(VerificationTable, self).to_dataframe(cubelist)
        self.ensure_all_pivot_columns(self.df)
        self.df.sort_index(axis=1, inplace=True)
