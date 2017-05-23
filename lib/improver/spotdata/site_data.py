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
Plugins written for the Improver site specific process chain.

"""

import numpy as np
import cPickle
from collections import OrderedDict


class ImportSiteData(object):
    '''
    Create a dictionary of site information from a variety of sources.
    Currently supported are the import of a pickle file with site
    information - called with 'pickle_file'
    Or lists of properties for sites - called with 'runtime_list'.

    '''

    def __init__(self, source):
        '''
        Class is called with the desired source of site data. The source
        may be a pickle file or a runtime_list that is defined on the command
        line or in the suite.

        Args:
        -----
        source : string setting the source of site data.

        '''
        self.source = source
        self.latitudes = None
        self.longitudes = None
        self.altitudes = None
        self.site_ids = None
        self.gmtoffsets = None

    def process(self, *args, **kwargs):
        '''Call the required method'''
        function = getattr(self, self.source)
        return function(*args, **kwargs)

    def pickle_file(self, file_path):
        '''
        Use a pickle file produced by the current SSPS system.

        '''
        site_data = self.read_pickle_file(file_path)

        self.latitudes = np.array([site.latitude for site in site_data])
        self.longitudes = np.array([site.longitude for site in site_data])
        self.altitudes = np.array([site.altitude for site in site_data])
        self.site_ids = np.array([site.bestdata_id for site in site_data])
        self.gmtoffsets = np.array([site.gmtoffset for site in site_data])

        return self.construct_site_dictionary()

    @staticmethod
    def read_pickle_file(file_path):
        '''
        Uses existing bestdata site routines to decode pickle file created
        by bestdata2.

        Args:
        -----
        file_path : Path to target pickle file.

        Returns:
        --------
        bd_site_data : bestdata site class containing site information.

        '''
        try:
            with open(file_path, 'rb') as bd_pickle_file:
                _ = cPickle.load(bd_pickle_file)
                [_, _, bd_site_data, _, _] = (cPickle.load(bd_pickle_file))
        except:
            raise Exception("Unable to read pickle file.")

        return bd_site_data

    def runtime_list(self, latitudes, longitudes,
                     altitudes=None, site_ids=None):
        '''
        Use data provided on the command line/controlling suite at runtime.

        '''
        if site_ids is not None:
            self.site_ids = np.array(site_ids)
        else:
            self.site_ids = np.arange(len(latitudes))
        if altitudes is not None:
            self.altitudes = np.array(altitudes)
        else:
            self.altitudes = np.zeros(len(latitudes))

        self.latitudes = np.array(latitudes)
        self.longitudes = np.array(longitudes)
        self.gmtoffsets = set_gmt_offset(self.longitudes)

        return self.construct_site_dictionary()

    def construct_site_dictionary(self):
        '''
        Constructs a dictionary of site data regardles of source to give the
        spotdata routines a consistent source of site data.

        Returns:
        --------
        sites : Dictionary of site data.

        '''
        sites = OrderedDict()
        for i_site, site_id in enumerate(self.site_ids):
            if self.gmtoffsets[i_site] is None:
                self.gmtoffsets[i_site] = 0
            sites.update(
                {site_id: {
                    'latitude': self.latitudes[i_site],
                    'longitude': self.longitudes[i_site],
                    'altitude': self.altitudes[i_site],
                    'gmtoffset': self.gmtoffsets[i_site]
                    }
                 })
        return sites


def set_gmt_offset(longitudes):
    '''
    Simplistic timezone setting for unset sites that uses 15 degree bins
    centred on 0 degrees longitude. Used for on the fly site generation
    when no more rigorous source of timeszone information is provided.

    Args:
    -----
    longitudes : list of longitudes.

    Returns:
    --------
    gmtoffsets : list of gmtoffsets calculated using longitude.

    '''
    return ((longitudes + (7.5*np.sign(longitudes)))/15).astype(int)
