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
"""Convert NetCDF files to realizations."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(cube: cli.inputcube,
            *,
            no_of_realizations: int = None,
            ecc_bounds_warning=False):
    """Converts an incoming cube into one containing realizations.

    Args:
        cube (iris.cube.Cube):
            A cube to be processed.
        no_of_realizations:
            The number of percentiles to be generated. This is equal to the
            number of ensemble realizations that will be generated.
        ecc_bounds_warning:
            If True, where percentiles exceed the ECC bounds range, raises a
            warning rather than an exception.

    Returns:
        cube (iris.cube.Cube):
            The processed cube.
    """
    from improver.cli import (percentiles_to_realizations,
                              probabilities_to_realizations, extract)

    if cube.coords('percentile'):
        cube = percentiles_to_realizations.process(
            cube, no_of_percentiles=no_of_realizations,
            rebadging=True, ecc_bounds_warning=ecc_bounds_warning)
    elif cube.coord(var_name='threshold'):
        cube = probabilities_to_realizations.process(
            cube, no_of_realizations=no_of_realizations, rebadging=True,
            ecc_bounds_warning=ecc_bounds_warning)
    return cube
