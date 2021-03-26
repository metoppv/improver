# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2021 Met Office.
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
"""Script to output metadata interpretation against Met Office IMPROVER standard"""

import argparse

from iris import load

from improver.developer_tools.metadata_interpreter import (
    MOMetadataInterpreter,
    display_interpretation,
)


def main(filepath, verbose=False):
    """
    Read file at filepath and write out metadata interpretation according to
    Met Office standard.  Optional flag to specify source of interpretation.
    This script is intended as a debugging tool to aid developers in adding
    and modifying metadata within the code base.

    Args:
        filepath (str):
            Full path to input NetCDF file
        verbose (bool):
            Boolean flag to output information about sources of metadata
            interpretation

    Raises:
        ValueError: various: if metadata are internally inconsistent or do not
            match the expected standard
    """
    cubes = load(filepath)

    for cube in cubes:
        interpreter = MOMetadataInterpreter()
        interpreter.run(cube)
        output = display_interpretation(interpreter, verbose=verbose)
        print(output)


if __name__ == "__main__":
    """Parse arguments and call"""
    parser = argparse.ArgumentParser(description="Interpret metadata")
    parser.add_argument("filepath", help="Full path to input NetCDF file")
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Boolean flag to output information about sources of metadata interpretation",
    )
    args = parser.parse_args()
    main(args.filepath, verbose=args.verbose)
