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

"""
Plugins written for the Improver site specific process chain.
For reading data processed by IMPROVER, and site specification input.

"""
import os
import iris
from improver.utilities.load import load_cubelist


def get_method_prerequisites(method, diagnostic_data_path):
    """
    Determine which additional diagnostics are required for a given
    method of data extraction.

    Args:
        method (string):
            String representing the method of data extraction that is being
            used.

    Returns:
        additional_diagnostics (dict):
            A dictionary keyed with the diagnostic names and containing the
            additional iris.cube.Cubes that are required.

    """

    prereq = {
        'model_level_temperature_lapse_rate': ['temperature_on_height_levels',
                                               'pressure_on_height_levels',
                                               'surface_pressure']
        }

    additional_diagnostics = {}
    if method in list(prereq.keys()):
        for item in prereq[method]:
            additional_diagnostics[item] = get_additional_diagnostics(
                item, diagnostic_data_path)

        return additional_diagnostics

    return None


def get_additional_diagnostics(diagnostic_name, diagnostic_data_path,
                               time_extract=None):
    """
    Load additional diagnostics needed for particular spot data processes.

    Args:
        diagnostic_name (string):
            The name of the diagnostic to be loaded. Used to find
            the relevant file.

        time_extract (iris.Constraint):
            An iris constraint to extract and return only data from the desired
            time.

    Returns:
        cube (iris.cube.CubeList):
            CubeList containing the desired diagnostic data, with a single
            entry if time_extract is provided.

    Raises:
        IOError : If files are not found.
        ValueError : If required diagnostics are not found in the read files.

    """
    # Search diadnostic data directory for all files relevant to current
    # diagnostic.
    files_to_read = [
        os.path.join(dirpath, filename)
        for dirpath, _, files in os.walk(diagnostic_data_path)
        for filename in files if diagnostic_name in filename]

    if not files_to_read:
        raise IOError('The relevant data files for {}, which is required '
                      'as an additional diagnostic is not available '
                      'in {}.'.format(
                          diagnostic_name, diagnostic_data_path))
    cubes = load_cubelist(files_to_read)

    if time_extract is not None:
        cubes = cubes.extract(time_extract)
        if not cubes:
            raise ValueError('No diagnostics match {}'.format(time_extract))
    return cubes
