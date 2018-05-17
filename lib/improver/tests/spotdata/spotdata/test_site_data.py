# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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
"""Unit tests for the spotdata.site_data"""

import unittest
import numpy as np
import json
from subprocess import call as Call
from tempfile import mkdtemp
from iris.tests import IrisTest

from improver.spotdata.site_data import ImportSiteData as Plugin


class Test_ImportSiteData(IrisTest):

    """
    Test the construction of SpotData site dictionaries from files and command
    line definitions.

    """

    def setUp(self):
        """Create components required for testing site_data."""

        self.data_directory = mkdtemp()
        self.site_data = [
            {'latitude': -51.6927, 'wmo_id': None, 'altitude': 0,
             'gmtoffset': -3.0, 'longitude': -57.8557},
            {'latitude': -47.24417, 'wmo_id': 15, 'altitude': 205,
             'gmtoffset': -4.0, 'longitude': -72.585},
            {'latitude': -40.35, 'wmo_id': 10, 'altitude': 54,
             'gmtoffset': 0.0, 'longitude': -9.8833},
            {'latitude': -33.8, 'wmo_id': 17, 'altitude': 25,
             'gmtoffset': 2.0, 'longitude': 18.37},
            {'latitude': -27.45, 'wmo_id': 18, 'altitude': 38,
             'gmtoffset': 10.0, 'longitude': 153.03}
            ]

        self.latitudes = [site['latitude'] for site in self.site_data]
        self.longitudes = (
            [site['longitude'] for site in self.site_data])
        self.altitudes = [site['altitude'] for site in self.site_data]
        self.wmo_sites = [0, 15, 10, 17, 18]
        self.variables = ['latitude', 'longitude', 'altitude', 'utc_offset']

    def tearDown(self):
        """Remove temporary directories created for testing."""
        Call(['rm', '-f', self.data_directory + '/site_data.json'])
        Call(['rmdir', self.data_directory])

    @staticmethod
    def change_key(list_of_dictionaries, oldkey, newkey):
        """
        Substitute a dictionary key in a list of dictionaries.

        """
        for dictionary in list_of_dictionaries:
            dictionary[newkey] = dictionary.pop(oldkey)

    def save_json(self, data):
        """
        Save the passed in data as a json file.

        """
        self.site_path = self.data_directory + '/site_data.json'
        ff = open(self.site_path, 'w')
        json.dump(data, ff, sort_keys=True, indent=4,
                  separators=(',', ': ',))
        ff.close()


class Test_from_file(Test_ImportSiteData):
    """Test function used for loading site data from json files."""

    method = 'from_file'

    def test_complete(self):
        """
        Test loading a valid json file containing site details.
        File formatted to represent current BestData input.

        """
        expected = self.site_data
        self.save_json(self.site_data)

        self.change_key(expected, 'gmtoffset', 'utc_offset')
        result = Plugin(self.method).process(self.site_path)

        for key in self.variables:
            expected_vals = np.array([site[key] for site in expected])
            result_vals = np.array([result[index][key] for index in result])
            self.assertArrayEqual(expected_vals, result_vals)

        # check that wmo sites are correctly flagged.
        self.assertArrayEqual(self.wmo_sites,
                              [result[index]['wmo_site'] for index in result])

    def test_no_latitude(self):
        """
        Test loading a valid json file containing site details,
        but missing compulsary latitude data.

        """
        for site in self.site_data:
            site.pop('latitude')
        self.save_json(self.site_data)

        msg = 'longitude and latitude must be defined'
        with self.assertRaisesRegex(KeyError, msg):
            Plugin(self.method).process(self.site_path)

    def test_no_longitude(self):
        """
        Test loading a valid json file containing site details,
        but missing compulsary longitude data.

        """
        for site in self.site_data:
            site.pop('longitude')
        self.save_json(self.site_data)

        msg = 'longitude and latitude must be defined'
        with self.assertRaisesRegex(KeyError, msg):
            Plugin(self.method).process(self.site_path)

    def test_unequal_lat_lon(self):
        """
        Test loading a valid json file containing site details,
        but missing compulsary longitude data for some sites.

        """
        self.site_data[0].pop('longitude')
        self.save_json(self.site_data)
        msg = 'Unequal no. of latitudes (.*) and longitudes'
        with self.assertRaisesRegex(ValueError, msg):
            Plugin(self.method).process(self.site_path)

    def test_only_lat_lon(self):
        """
        Test loading a valid json file containing site details, only containing
        lat/lon. Altitudes set to fill value of np.nan, and utc_offset
        calculated from longitude.

        """
        for site in self.site_data:
            site.pop('wmo_id')
            site.pop('altitude')
            site.pop('gmtoffset')

        self.save_json(self.site_data)
        result = Plugin(self.method).process(self.site_path)
        expected_altitudes = [np.nan] * 5
        expected_utc_offsets = [-4, -5, -1, 1, 10]

        self.assertArrayEqual(
            expected_altitudes,
            np.array([result[index]['altitude'] for index in result]))
        self.assertArrayEqual(
            expected_utc_offsets,
            np.array([result[index]['utc_offset'] for index in result]))


class Test_runtime_list(Test_ImportSiteData):
    """
    Test setting up site data at run time through the CLI.

    """
    method = 'runtime_list'

    def test_complete(self):
        """
        Test creating sites at runtime. Here latitude, longitude, and altitude
        are provided.

        """
        site_data = self.site_data
        # These are not currently a CLI option, so remove them.
        for site in site_data:
            site.pop('gmtoffset')
            site.pop('wmo_id')

        result = Plugin(self.method).process(
            site_properties=site_data)
        # set from longitude.
        expected_utc_offsets = [-4, -5, -1, 1, 10]
        # unset, so none can be flagged as wmo_sites
        expected_wmo_sites = np.zeros(len(result))

        self.assertArrayEqual(
            self.latitudes,
            np.array([result[index]['latitude'] for index in result]))
        self.assertArrayEqual(
            self.longitudes,
            np.array([result[index]['longitude'] for index in result]))
        self.assertArrayEqual(
            self.altitudes,
            np.array([result[index]['altitude'] for index in result]))
        self.assertArrayEqual(
            expected_wmo_sites,
            np.array([result[index]['wmo_site'] for index in result]))
        self.assertArrayEqual(
            expected_utc_offsets,
            np.array([result[index]['utc_offset'] for index in result]))

    def test_only_lat_lon(self):
        """
        Test creating sites at runtime. Here only latitude and longitude are
        provided.

        """
        site_data = self.site_data
        for site in site_data:
            site.pop('gmtoffset')
            site.pop('wmo_id')
            site.pop('altitude')

        result = Plugin(self.method).process(site_data)
        expected_altitudes = [np.nan] * 5
        expected_utc_offsets = [-4, -5, -1, 1, 10]
        # unset, so none can be flagged as wmo_sites
        expected_wmo_sites = np.zeros(len(result))

        self.assertArrayEqual(
            self.latitudes,
            np.array([result[index]['latitude'] for index in result]))
        self.assertArrayEqual(
            self.longitudes,
            np.array([result[index]['longitude'] for index in result]))
        self.assertArrayEqual(
            expected_altitudes,
            np.array([result[index]['altitude'] for index in result]))
        self.assertArrayEqual(
            expected_utc_offsets,
            np.array([result[index]['utc_offset'] for index in result]))
        self.assertArrayEqual(
            expected_wmo_sites,
            np.array([result[index]['wmo_site'] for index in result]))


if __name__ == '__main__':
    unittest.main()
