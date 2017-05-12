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
This module defines the utilities required for Ensemble Copula Coupling
plugins.

"""
import copy
import numpy as np
import random

import cf_units as unit
import iris
from iris.exceptions import CoordinateNotFoundError


def create_percentiles(no_of_percentiles, sampling="quantile"):
    """
    Function to create percentiles.

    Parameters
    ----------
    no_of_percentiles : Int
        Number of percentiles.
    sampling : String
        Type of sampling of the distribution to produce a set of
        percentiles e.g. quantile or random.
        Accepted options for sampling are:
        Quantile: A regular set of equally-spaced percentiles aimed
                  at dividing a Cumulative Distribution Function into
                  blocks of equal probability.
        Random: A random set of ordered percentiles.

    For further details, Flowerdew, J., 2014.
    Calibrating ensemble reliability whilst preserving spatial structure.
    Tellus, Series A: Dynamic Meteorology and Oceanography, 66(1), pp.1-20.
    Schefzik, R., Thorarinsdottir, T.L. & Gneiting, T., 2013.
    Uncertainty Quantification in Complex Simulation Models Using Ensemble
    Copula Coupling.
    Statistical Science, 28(4), pp.616-640.

    Returns
    -------
    percentiles : List
        Percentiles calculated using the sampling technique specified.

    """
    if sampling in ["quantile"]:
        percentiles = np.linspace(
            1/float(1+no_of_percentiles),
            no_of_percentiles/float(1+no_of_percentiles),
            no_of_percentiles).tolist()
    elif sampling in ["random"]:
        percentiles = []
        for _ in range(no_of_percentiles):
            percentiles.append(
                random.uniform(
                    1/float(1+no_of_percentiles),
                    no_of_percentiles/float(1+no_of_percentiles)))
        percentiles = sorted(percentiles)
    else:
        msg = "The {} sampling option is not yet implemented.".format(
            sampling)
        raise ValueError(msg)
    return percentiles


def create_cube_with_percentiles(percentiles, template_cube, cube_data):
    """
    Create a cube with a percentile coordinate based on a template cube.

    Parameters
    ----------
    percentiles : List
        Ensemble percentiles.
    template_cube : Iris cube
        Cube to copy majority of coordinate definitions from.
    cube_data : Numpy array
        Data to insert into the template cube.
        The data is expected to have the shape of
        percentiles (0th dimension), time (1st dimension),
        y_coord (2nd dimension), x_coord (3rd dimension).

    Returns
    -------
    String
        Coordinate name of the matched coordinate.

    """
    def _append_to_aux_coords_and_dims(coord_name, aux_coords_and_dims):
        """
        Try to append a tuple containing a desired auxiliary coordinate,
        if the auxiliary coordinate is present on the template cube.

        Parameters
        ----------
        coord_name : String
            The name of the desired auxiliary coordinate.
        aux_coords_and_dims : List of tuples
            List of format: [(aux_coord1, dim_coord_to_be_associated_with),
                             (aux_coord2, dim_coord_to_be_associated_with)]
            For example: [(forecast_period, 1), (forecast_reference_time, 1)]

        """
        try:
            coord = template_cube.coord(coord_name)
            for coord_tuple in dim_coords_and_dims:
                if coord_tuple[0].name() in ["time"]:
                    time_dim = coord_tuple[1]
                    break
            aux_coords_and_dims.append((coord, time_dim))
        except CoordinateNotFoundError:
            pass

    percentile_coord = iris.coords.DimCoord(
        np.float32(percentiles), long_name="percentile",
        units=unit.Unit("1"), var_name="percentile")

    # Aim to create a list of tuples for setting the dim_coords_and_dims
    # required for a cube. The "realization" or "probability_above_threshold"
    # coordinates on the cube are ignored, with all other coordinates being
    # added to the dim_coords_and_dims list. The percentile coordinate tuple
    # is prepended to this list.
    dim_coords = []
    dims = []
    index = 1
    for coord in template_cube.dim_coords:
        if coord.name() in ["realization", "probability_above_threshold"]:
            continue
        dim_coords.append(coord)
        dims.append(index)
        index += 1

    dim_coords_and_dims = []
    for coord, dim in zip(dim_coords, dims):
        dim_coords_and_dims.append((coord, dim))
    dim_coords_and_dims = [(percentile_coord, 0)] + dim_coords_and_dims

    aux_coords_and_dims = []
    _append_to_aux_coords_and_dims(
        "forecast_reference_time", aux_coords_and_dims)
    _append_to_aux_coords_and_dims(
        "forecast_period", aux_coords_and_dims)

    metadata_dict = copy.deepcopy(template_cube.metadata._asdict())

    cube = iris.cube.Cube(
        cube_data, dim_coords_and_dims=dim_coords_and_dims,
        aux_coords_and_dims=aux_coords_and_dims, **metadata_dict)
    cube.attributes = template_cube.attributes
    cube.cell_methods = template_cube.cell_methods
    return cube
