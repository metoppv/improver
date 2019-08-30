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
from enum import Enum

from improver.utilities.cli_utilities import load_json_or_none
from improver.utilities.save import save_netcdf
from improver.utilities.load import load_cube, load_cubelist


def call_all(args, process_function, save_name, files):
    """A function to load cubes, run function and save the cubes.

    It starts by copying the ArgParser dictionary and removing the 'profile'
    and 'profile_file' keys.
    It checks that the extra strings given exists in the argParser.
    It the removes the save filepaths from the arguments dicts and stores them
    for later.
    Sends to cubes and json off to be loaded and updated the dictionary.
    Runs with function with all the dicts values as positional args.

    Args:
        args (improver.argparser.ArgParser):
            An argParser with the given arguments.
        process_function:
            The process function to be used.
        save_name (list of str):
            A list of the ArgParser names for save files.
        cube_args (list of str, optional):
            A list of the ArgParser names for positional filepath.
        cubelist_args (list of str, optional):
            A list of argParser names for cubelists.
        option_cube_args (list of str, optional):
            A list of the ArgParser name for keyword arguments.
        json_args (list of str, optional):

    Returns:
        (None)

    """
    d = vars(args)
    _ = [d.pop(x) for x in ['profile', 'profile_file']]
    # TODO does this make codacy happy?
    del _
    save = [d.pop(x) for x in save_name]

    for k, v in files.items():
        if v == FileType.CUBE:
            print('*' * 80)
            print("k: {}, v: {}".format(k, v))
            d[k] = load_cube(d[k])
        if v == FileType.OPTIONAL_CUBE:
            d[k] = load_cube(d[k], allow_none=True)
        if v == FileType.CUBELIST:
            d[k] = load_cubelist(d[k])
        if v == FileType.JSON:
            d[k] = load_json_or_none(d[k])

    result = process_function(*d.values())
    # TODO test this works with a tuple returning cli.
    if isinstance(result, tuple):
        for res, fpath in zip(result, save):
            save_netcdf(res, fpath)
    else:
        save_netcdf(result, save[0])
    return result


class FileType(Enum):
    CUBE = 1
    OPTIONAL_CUBE = 2
    CUBELIST = 3
    JSON = 4
