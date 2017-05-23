"""
Plugins written for the Improver site specific process chain.

"""

import os
from iris import FUTURE

FUTURE.netcdf_no_unlimited = True


class WriteOutput(object):
    ''' Writes diagnostic cube data in a format determined by the method.'''

    def __init__(self, method):
        '''
        Select the method (format) for writing out the data cubes.

        Args:
        -----
        method : String that sets the method.

        '''
        self.method = method
        self.dir_path = os.path.dirname(os.path.realpath(__file__))

    def process(self, cube):
        '''Call the required method'''
        function = getattr(self, self.method)
        function(cube)

    def as_netcdf(self, cube, path=None):
        '''
        Writes iris.cube.Cube data out to netCDF format files.

        Args:
        -----
        cube : iris.cube.Cube diagnostic data cube.
        path : Optional string setting the output path for the file.

        Returns:
        --------
        Nil. Writes out file to filepath or working directory.

        '''
        from iris.fileformats.netcdf import Saver
        if path is None:
            path = self.dir_path
        with Saver('{}/{}.nc'.format(path, cube.name()), 'NETCDF4') as output:
            output.write(cube)
