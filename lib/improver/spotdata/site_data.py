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
import json
from collections import OrderedDict


class ImportSiteData(object):
    """
    Create a dictionary of site information from a variety of sources.
    Currently supported are the import of a json file with site
    information - called with 'from_file'.
    Or lists of properties for sites - called with 'runtime_list'.

    """

    def __init__(self, source):
        """
        Class is called with the desired source of site data. The source
        may be a pickle file or a runtime_list that is defined on the command
        line or in the suite.

        Args:
            source (string):
                String setting the source of site data, available options are:
                - 'from_file' to read in site specifications from a file.
                - 'runtime_list' to interpret lists passed to the CLI as site
                  definitions.

        """
        self.source = source
        if self.source == 'from_file':
            self.function = self.from_file
        elif self.source == 'runtime_list':
            self.function = self.parse_input
        else:
            raise AttributeError('Unknown method "{}" passed to {}.'.format(
                self.source, self.__class__.__name__))

        self.latitudes = None
        self.longitudes = None
        self.altitudes = None
        self.utc_offsets = None
        self.wmo_site = None

    def __repr__(self):
        """Represent the configured plugin instance as a string."""
        result = ('<ImportSiteData: source: {}>')
        return result.format(self.source)

    def process(self, filepath=None, site_properties=None):
        """Call the required method."""
        args = [arg for arg in [filepath, site_properties] if arg is not None]
        return self.function(*args)

    def from_file(self, file_path):
        """
        Calls the correct reading function based on file type. Allows for
        different file types to be easily introduced.

        Args:
            file_path (string):
                Path to file containing SpotData site information.

        Raises:
            IOError : For unknown file types.

        """
        if file_path.endswith('.json'):
            data_file = open(file_path, 'r')
            site_data = json.load(data_file)
            data_file.close()
        else:
            raise IOError('Unknown file type for site definitions.')

        return self.parse_input(site_data)

    def parse_input(self, site_data):
        """
        Perform checks on sitedata input and use it to create a standard format
        sites dictionary.

        Args:
            site_data (dictionary):
                A dictionary of site data for checking and use in constructing
                a consistently formatted site dictionary. This step is to allow
                for input from either a read in file or for definition on the
                command line.

                Contains : Required

                latitudes/longitudes (lists of floats):
                    Lists of coordinates at which to extract data, these define
                    the positions of the SpotData sites.

                Contains : Optional

                altitudes (list of floats):
                    List of altitudes that can optionally be used in defining
                    sites on the fly. If unset the altitudes will be assumed as
                    equivalent to which ever neighbouring grid point is used as
                    a data source.

                utc_offsets (list of floats):
                    List of utc_offsets for calculating local time of the site.

        Returns:
            sites (dict):
                Dictionary of site data.

        Raises:
            KeyError : If longitude or latitude information is not found.

        """
        latitude_entries = [i_site for (i_site, site) in enumerate(site_data)
                            if 'latitude' in site.keys()]
        longitude_entries = [i_site for (i_site, site) in enumerate(site_data)
                             if 'longitude' in site.keys()]

        if not latitude_entries or not longitude_entries:
            raise KeyError('longitude and latitude must be defined for '
                           'site in site_data file')

        # Raise an error if there are an unequal number of latitudes and
        # longitudes as it is indicative of an error in the input data.
        if latitude_entries != longitude_entries:
            raise ValueError(
                'Unequal no. of latitudes ({}) and longitudes ({}).'.format(
                    len(latitude_entries), len(longitude_entries)))
        else:
            valid_entries = latitude_entries

        site_data = [site_data[i_site] for i_site in valid_entries]

        self.latitudes = np.array([site['latitude'] for site in site_data])
        self.longitudes = np.array([site['longitude'] for site in site_data])

        n_sites = len(self.latitudes)

        # If altitudes are unset use np.nan to indicate that they are at the
        # altitude of their neighbouring grid point. Likewise, if sites are
        # wmo sites set wmo_site to wmo_id, else set it to 0.
        self.altitudes = np.full(n_sites, np.nan)
        self.wmo_site = np.full(n_sites, 0, dtype=int)
        for i_site, site in enumerate(site_data):
            if 'altitude' in site.keys() and site['altitude'] is not None:
                self.altitudes[i_site] = site['altitude']
            if 'wmo_id' in site.keys() and site['wmo_id'] is not None:
                self.wmo_site[i_site] = site['wmo_id']

        # Identify UTC offset if it is provided in the input, otherwise set it
        # based upon site longitude.
        self.utc_offsets = np.full(n_sites, np.nan)
        for i_site, site in enumerate(site_data):
            if 'gmtoffset' in site.keys():
                self.utc_offsets[i_site] = site['gmtoffset']
            elif 'utcoffset' in site.keys():
                self.utc_offsets[i_site] = site['utcoffset']
            elif 'utc_offset' in site.keys():
                self.utc_offsets[i_site] = site['utc_offset']

            # If it's not been set, set it with the longitude based method.
            if np.isnan(self.utc_offsets[i_site]):
                self.utc_offsets[i_site], = set_utc_offset([site['longitude']])

        return self.construct_site_dictionary()

    def construct_site_dictionary(self):
        """
        Constructs a dictionary of site data regardless of source to give the
        spotdata routines a consistent source of site data.

        Returns:
            sites (dict):
                Dictionary of site data.

        """
        sites = OrderedDict()
        for i_site, _ in enumerate(self.latitudes):
            sites[i_site] = {
                'latitude': self.latitudes[i_site],
                'longitude': self.longitudes[i_site],
                'altitude': self.altitudes[i_site],
                'utc_offset': self.utc_offsets[i_site],
                'wmo_site': self.wmo_site[i_site]
                }

        return sites


def set_utc_offset(longitudes):
    """
    Simplistic timezone setting for unset sites that uses 15 degree bins
    centred on 0 degrees longitude. Used for on the fly site generation
    when no more rigorous source of timeszone information is provided.

    Args:
        longitudes (List):
            List of longitudes.

    Returns:
        utc_offsets (List):
            List of utc_offsets calculated using longitude.

    """
    return np.floor((np.array(longitudes) + 7.5)/15.)
