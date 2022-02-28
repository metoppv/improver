#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2022 Met Office.
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
"""Script to interpret the metadata of an IMPROVER output."""

from improver import cli


@cli.clizefy
def process(*file_paths: cli.inputpath, verbose=False, failures_only=False):
    """Intepret the metadata of an IMPROVER output into human readable format
    according to the IMPROVER standard. An optional verbosity flag, if set to
    True, will specify the source of each interpreted element.

    This tool is intended as an aid to developers in adding and modifying
    metadata within the code base.

    Args:
        file_paths (list of Path objects):
            File paths to netCDF files for which the metadata should be interpreted.
        verbose (bool):
            Boolean flag to output information about sources of metadata
            interpretation.
        failures_only (bool):
            Boolean flag that, if set, means only information about non-compliant
            files is printed.
    Raises:
        ValueError: If any of the input files are not metadata compliant.
    """
    from iris import load

    from improver.developer_tools.metadata_interpreter import (
        MOMetadataInterpreter,
        display_interpretation,
    )

    cubelists = {file_path: load(file_path.as_posix()) for file_path in file_paths}

    any_failures = False
    for file, cubelist in cubelists.items():
        for cube in cubelist:
            interpreter = MOMetadataInterpreter()
            try:
                interpreter.run(cube)
            except ValueError as err:
                output = "Non-compliant :\n{}".format(str(err))
                any_failures = True
            else:
                if failures_only:
                    continue
                output = display_interpretation(interpreter, verbose=verbose)
            print(f"\nfile : {file}")
            print(f"cube name : {cube.name()}")
            print(output)

    if any_failures:
        raise ValueError("One or more files checked is not metadata compliant")
