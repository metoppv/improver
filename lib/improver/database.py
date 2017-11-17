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
import numpy as np
import sqlite3
import os
from datetime import datetime as dt
from pandas import DataFrame


class SpotDatabase(object):
    """
    Class to create a Database table from a SpotData iris.cube.

    """

    def __init__(self, output, outfile, tablename,
                 extra_columns=None,
                 extra_values=None):

        """
        Initialise class.

        """

        self.output = output
        self.outfile = outfile
        self.tablename = tablename

        if extra_columns and extra_values:
            self.column_dims = self.column_dims + [extra_columns]
            self.column_maps = self.column_maps + [extra_values]

    def __repr__(self):
        """
        Representation of the instance.

        """
        result = ('<SpotDatabase: columns (primary={primary_map}, '
                  'other={column_dims}, '
                  'pivot={pivot_dim})>')
        return result.format(**self.__dict__)

    def to_dataframe(self, cubelist):
        """
        Turn the input cubes into a 2-dimensional DataFrame object

        """

        for cube in cubelist:
            for c in cube.slices_over(self.column_dims[0]):
                df = DataFrame(c.data,
                               index=c.coord(self.primary_dim).points,
                               columns=['vals'])

                if self.pivot_dim:
                    # Reshape data based on column values
                    coords = c.coord(self.pivot_dim).points
                    col_names = map(self.pivot_map, coords)
                    df.insert(1, self.pivot_dim, col_names)
                    df = df.pivot(columns=self.pivot_dim, values='vals')

                if self.primary_map:
                    # Switch the index out for a map if specified
                    for mapping, function in zip(self.primary_map,
                                                 self.primary_func):
                        df.insert(0, mapping, map(function, df.index))

                    # Takes significant time if a multi-index
                    df.set_index(self.primary_map, inplace=True)

                for dim, col in itertools.izip_longest(self.column_dims,
                                                       self.column_maps):
                    if dim in df.columns:
                        continue

                    if dim in [coord.standard_name or coord.long_name
                               for coord in c.dim_coords + c.aux_coords]:
                        coord = c.coord(dim)
                        column_name = col or dim
                        if len(coord.points) == 1:
                            column_data = coord.points[0]
                        else:
                            column_data = coord.points

                    # Check to see if provided dim is a method of the cube
                    elif hasattr(c, dim):
                        attr = getattr(c, dim)
                        if callable(attr):
                            column_name = col
                            column_data = attr()

                    else:
                        column_name = col
                        column_data = dim

                    as_index = True if dim != self.pivot_dim else False
                    self.insert_column(df, column_name, column_data, as_index)

                try:
                    self.df = self.df.combine_first(df)
                except AttributeError:
                    # Ensure the first instance has all pivot columns created
                    self.ensure_all_pivot_columns(df)
                    self.df = df

    @staticmethod
    def insert_column(dataframe, column_name, column_data, as_index=False):
        """
        Method to insert a column at front of a datframe, adding to the index
        if required, creating a multiple-index dataframe.
        """
        dataframe.insert(1, column_name, column_data)
        if as_index:
            dataframe.set_index(column_name, append=True, inplace=True)

    def ensure_all_pivot_columns(self, dataframe):
        """
        Method to ensure all pivot columns exist in the dataframe, adding any
        that do not.

        """

        if self.pivot_dim and self.pivot_max:
            for pivot_val in range(self.pivot_max+1):
                if not self.pivot_map(pivot_val) in dataframe.columns:
                    dataframe[self.pivot_map(pivot_val)] = np.nan

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

    def to_csv(self, outfile):
        """
        Output the dataframe to comma seperated file.

        """

        self.df.to_csv(outfile)

    def process(self, cubelist):
        """
        Method to perform the table creation and output to file.

        """

        self.to_dataframe(cubelist)

        if self.output == 'sqlite':
            self.to_sql(self.outfile, self.tablename)

        if self.output == 'csv':
            self.to_csv(self.outfile)


class VerificationTable(SpotDatabase):
    """
    Represents a single Verification database table

    """

    def __init__(self, *args, **kwargs):

        self.primary_dim = 'time'
        self.primary_map = ['validity_date', 'validity_time']
        self.primary_func = [lambda x:dt.utcfromtimestamp(x).date(),
                             lambda x:dt.utcfromtimestamp(x).hour*100]

        self.pivot_dim = 'forecast_period'
        self.pivot_map = lambda x: 'fcr_tplus{:03d}'.format(int(x/3600))
        self.pivot_max = 54*60*60

        self.column_dims = ['wmo_site', 'name']
        self.column_maps = ['station_id', 'cf_name']

        super(VerificationTable, self).__init__(*args, **kwargs)
