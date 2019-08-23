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
"""Pattern for calling clis"""
from improver.utilities.cli_utilities import load_json_or_none
from improver.utilities.save import save_netcdf
from improver.utilities.load import load_cube, load_cubelist


def call_all(args, process_function, save_name, cube_args=[],
             cubelist_args=[], option_cube_args=[], json_args=[]):
    """A function to load cubes, run function and save the cubes.

    It starts by copying the ArgParser dictionary and removing the 'profile'
    and 'profile_file' keys.
    It the removes the save filepaths from the arguments dicts and stores them
    for later.
    For loading cubes, it works through the list of cubes to load, gets the
    file path from the dictionary then replaces the filepath with the cube.

    cube_args = ['input']
    dict {'input': 'file/path'}
    the loads
    dict['input'] = load(dict['input']
    resulting in
    dict {'input': 'An iris.cube.Cube'}


    Args:
        cube_args (list of str):
            A list of the ArgParser names for positional filepath..
        option_cube_args (list of str):
            A list of the ArgParser name for keyword arguments.
        args (improver.argparser.ArgParser):
            An argParser with the given arguments.
        process_function:
            The process function to be used.
        save_name (list of str):
            A list of the ArgParser names for save files.

    Returns:
        None

    """
    d = args.__dict__.copy()
    [d.pop(x) for x in ['profile', 'profile_file']]
    for i in cube_args + cubelist_args + option_cube_args + save_name:
        if i not in d.keys():
            raise ValueError('Argument {} given not in argParser args. {}'.format(i, d))
    save = [d.pop(x) for x in save_name]

    for i in json_args:
        d[i] = load_json_or_none(d[i])
    for i in cube_args:
        d[i] = load_cube(d[i])
    for i in cubelist_args:
        d[i] = load_cubelist(d[i])
    for i in option_cube_args:
        d[i] = load_cube(d[i], allow_none=True)

    result = process_function(*d.values())
    for file in save:
        save_netcdf(result, file)

