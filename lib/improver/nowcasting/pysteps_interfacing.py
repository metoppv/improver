# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2019 Met Office.
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
Functions to assist with interface to pysteps library
"""

import numpy as np
from numpy.ma import MaskedArray

import iris


def pysteps_importer(precip_cube):
    """
    Translate a precipitation cube into the format required by pysteps
    optical flow and extrapolation algorithms

    Args:
        precip_cube (iris.cube.Cube):
            Cube of precipitation rates

    Returns:
        precip_rate (np.ndarray):
            2D array of precipitation rates
        metadata (dict):
            Dictionary of metadata required by pysteps algorithms

    Reference:
        https://pysteps.readthedocs.io/en/latest/pysteps_reference/
        io.html#pysteps-io-importers
    """

    pass


def pysteps_exporter(data, metadata, template_cube):
    """
    Translate the data output from a pysteps algorithm into IMPROVER output
    format

    Args:
        data (np.ndarray):
            Data to be written
        metadata (dict):
            Dictionary of metadata (as input to pysteps algorithm??)
        template_cube (iris.cube.Cube):
            Cube containing coordinate information and metadata for output

    Returns:
        data_cube (iris.cube.Cube):
            Input data packaged in IMPROVER format cube, with correct
            datatypes and units
    """

    pass

