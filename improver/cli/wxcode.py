#!/usr/bin/env python
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
"""CLI to generate weather symbols."""

from improver import cli


def _extend_help(fn):
    # TODO: speed up help - pulling in decision tree imports iris
    # (and gets executed at import time)
    from improver.wxcode.utilities import interrogate_decision_tree
    for wxtree in ('high_resolution', 'global'):
        title = wxtree.capitalize().replace('_', ' ') + ' tree inputs'
        inputs = interrogate_decision_tree(wxtree).replace('\n', '\n        ')
        tree_help = f"""
    {title}::

        {inputs}
    """
        fn.__doc__ += tree_help
    return fn


@cli.clizefy
@cli.with_output
@_extend_help
def process(*cubes: cli.inputcube,
            wxtree='high_resolution'):
    """ Processes cube for Weather symbols.

    Args:
        cubes (iris.cube.CubeList):
            A cubelist containing the diagnostics required for the
            weather symbols decision tree, these at co-incident times.
        wxtree (str):
            Weather Code tree: high_resolution or global.

    Returns:
        iris.cube.Cube:
            A cube of weather symbols.
    """
    from iris.cube import CubeList
    from improver.wxcode.weather_symbols import WeatherSymbols

    if not cubes:
        raise RuntimeError('Not enough input arguments. '
                           'See help for more information.')

    return WeatherSymbols(wxtree=wxtree).process(CubeList(cubes))
