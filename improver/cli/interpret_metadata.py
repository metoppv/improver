#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
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
