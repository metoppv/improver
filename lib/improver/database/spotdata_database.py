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

import iris
import iris.iterate
import iris.pandas
import itertools
import numpy as np
import pandas as pd
import sqlite3
import os
from datetime import datetime as dt
from pandas import DataFrame


# For debugging
from profiler import profile
import pprint
import resource




class SpotDatabase(object):
    
    def __init__(self, cubelist, 
                 primary_dim = 'time', 
                 primary_map = ['validity_date', 'validity_time'],
                 primary_func = [lambda x:dt.utcfromtimestamp(x).date(),
                                 lambda x:dt.utcfromtimestamp(x).hour*100],
                 
                 pivot_dim='forecast_period',
                 pivot_map=lambda x:'fcr_tplus{:03d}'.format(int(x/3600)),
                 
                 column_dims=['wmo_site', 'name'],
                 column_maps=['station_id', 'cf_name']):
        
        """
        Initialise class.
        
              
        """
        
        self.cubelist  = cubelist
        self.pivot_dim = pivot_dim
        self.pivot_map = pivot_map
        self.primary_dim = primary_dim
        self.primary_map  = primary_map
        self.primary_func = primary_func
        
        self.column_dims = column_dims
        self.column_maps = column_maps
        self.column_func = []
        
        
        self.assert_similar()

        
    def __repr__(self):
        
        """
        Representation of the instance.
        
        """
        
        
        return '<SpotDatabase: {}>'.format(self.primary_dim)
        
        
    def assert_similar(self):
        
        """
        Ensure that the dimensions and coordinates are shared between cubes.
        
        """
        cubelist = self.cubelist
        some_cube = self.cubelist[0]
        
        for cube in cubelist:
            for coord in cube.dim_coords:
                assert coord.is_compatible(some_cube.coord(coord.name()))
        
    
    def determine_dimensions(self, cube):
        
        """
        Determine the dimensions to collapse from the input cube.
        
        """
        
        dimensions = cube.dim_coords
        dim_names  = [dim.standard_name or dim.long_name for dim in dimensions]
        
        if [self.cols] + [self.rows] in dim_names:
            self.row.index = dim_names.index(self.row)
    
    #@profile
    def to_dataframe(self):
        """
        Turn the input cubes into a 2-dimensional DataFrame object
        
        """
        
        #coords   = [coord.standard_name or coord.long_name for coord in 
        #                                            self.cubelist[0].coords()]
        #ignore = [item for item in coords if item not in [self.column_dims] +
        #                                                 [self.primary_dim] +
        #                                                 [self.pivot_dim]]
        for cube in self.cubelist:
            print cube
            for coord in self.cubelist[0].coords():
                name = coord.standard_name or coord.long_name
                
                if (name not in self.column_dims + [self.primary_dim] +
                                                   [self.pivot_dim]):
                    cube.remove_coord(coord)

            # Loop over the remaining dimensions
            for c in cube.slices_over(1):
                cube_name = cube.name()
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
                
                
        #self.df = self.df.pivot(columns = 'forecast_period', values='vals')

        
    def constrained_to_dataframe():
        """
        Turn the input cubes into a dataframe sorted by validity_time
        
        """
        
        cubes = self.cubelist
        rows  = self.primary_dim
        coords   = [coord.standard_name or coord.long_name for coord in 
                                                    self.cubelist[0].coords()]
        for cube in cubes:
            ignored_coord_dims = [item for item in ignored_coords if item not in cube.dim_coords]
            print cube
            for c in cube.slices_over('time'):
                df = iris.pandas.as_data_frame(c, copy=False)
        
        primary_key_iterator = self.determine_range(rows)
        for row_val in sorted(primary_key_iterator, key=lambda x: x.points):
            print row_val
            records = []
            
            # Constrain the cubes by primary_dim
            constraint = iris.Constraint(**{rows : row_val.points})
            selection  = self.cubelist.extract(constraint)
            
            record  = dict()
            for cube in selection:
                df = iris.pandas.as_data_frame(cube, copy=False)
            return
            for iterator in iris.iterate.izip(*selection):
                df = iris.pandas.as_data_frame(cube, copy=False)
                for cube in iterator:
                    dictionary = cube.coords()
                    record = {coord.standard_name or
                              coord.long_name : coord.points[0]
                              for coord in [cube.coord(c) for c in cols]}
                    records.append(record)
            new_df = DataFrame.from_dict(records)
            self.df =  pd.concat([self.df, new_df])
        return self.df
        
    def create_table(self, outfile, table='test'):
        """
        Create the SQL datafile table 
        
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
        
    def determine_range(self, dimension):
    
        """
        Determine the unique values of the dimension over which to unroll into 
        primary key rows in the table.
        
        """
        
        unique_values = set()
        for cube in self.cubelist:
            for val in cube.coord(dimension):
                unique_values.add(val)
                
        return unique_values
        
