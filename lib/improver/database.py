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
import sqlite3
import os
from datetime import datetime as dt
from pandas import DataFrame

class SpotDatabase(object):
    """
    Class to create a Database table from a SpotData iris.cube.

    """

    def __init__(self, cubelist,
                 primary_dim='time',
                 primary_map=['validity_date', 'validity_time'],
                 primary_func=[lambda x:dt.utcfromtimestamp(x).date(),
                               lambda x:dt.utcfromtimestamp(x).hour*100],

                 pivot_dim='forecast_period',
                 pivot_map=lambda x: 'fcr_tplus{:03d}'.format(int(x/3600)),

                 column_dims=['wmo_site', 'name'],
                 column_maps=['station_id', 'cf_name']):

        """
        Initialise class.

        """

        self.cubelist = cubelist
        self.pivot_dim = pivot_dim
        self.pivot_map = pivot_map
        self.primary_dim = primary_dim
        self.primary_map = primary_map
        self.primary_func = primary_func

        self.column_dims = column_dims
        self.column_maps = column_maps
        self.column_func = []

        self.assert_similar()

    def __repr__(self):
        """
        Representation of the instance.

        """
        result = '<SpotDatabase: columns (primary={}, other={}, pivot={})>'
        return result.format(self.primary_map, self.column_maps, self.pivot_dim)

    def assert_similar(self):
        """
        Ensure that the dimensions and coordinates are shared between cubes.

        """
        cubelist = self.cubelist
        some_cube = self.cubelist[0]

        for cube in cubelist:
            for coord in cube.dim_coords:
                assert coord.is_compatible(some_cube.coord(coord.name()))

    def to_dataframe(self):
        """
        Turn the input cubes into a 2-dimensional DataFrame object

        """

        for cube in self.cubelist:
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

                    if dim in [coord.standard_name or coord.long_name for coord in
                                                  c.dim_coords + c.aux_coords]:
                        coord = c.coord(dim)
                        col_name = col or dim
                        if len(coord.points) == 1:
                            col_data = coord.points[0]
                        else:
                            col_data = coord.points
                    else:
                        # Should have a conditional
                        col_name = col or dim
                        col_data = cube.name()

                    df.insert(1, col_name, col_data)
                    if dim != self.pivot_dim:
                        # This is rather expensive
                        df.set_index(col_name, append=True, inplace=True)
                try:
                    self.df = self.df.combine_first(df)
                except AttributeError:
                    self.df = df

    def create_table(self, outfile, table='test'):
        """
        Method to first create a the SQL datafile table.
        
        The primary keys are determined from the indexed columns in
        the DataFrame.
        The SQLite3 table's datatypes are determined from the data types in
        the DataFrame.

        """

        if os.path.isfile(outfile):
            os.unlink(outfile)

        # Remove the current index, and use the indexed columns for for db keys
        columns = self.df.columns
        new_df = self.df.reset_index()
        n_keys = len(new_df) - len(columns)

        schema = pd.io.sql.get_schema(self.df, table,
                                      flavor='sqlite',
                                      keys=self.df.columns[:n_keys])

        with sqlite3.connect(outfile) as db:
            db.execute(schema)

    def to_sql(self, outfile, table='test', new=True):
        """
        Output the dataframe to SQL database file

        """

        with sqlite3.connect(outfile) as db:
            self.df.to_sql(table, con=db, if_exists='append', index=True)

    def to_csv(self, outfile):
        """
        Output the dataframe to comma seperated file

        """

        self.df.to_csv(outfile)
