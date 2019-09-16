#!/usr/bin/env bats
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

@test "cube-combiner -h" {
  PYTHONPATH=$PWD/lib python3 -m improver.cli combine -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
Usage: python3 -m improver.cli combine [OPTIONS] [cubelist...]

Module for combining Cubes.

Combine the input cubes into a single cube using the requested operation.  e.g. '+', '-', '*', 'add', 'subtract', 'multiply', 'min', 'max', 'mean'

Arguments:
  cubelist...                An iris CubeList to be combined. (type: INPUTCUBE)

Options:
  --operation=STR            "+", "-", "*", "add", "subtract", "multiply", "min", "max", "mean" An operation to use in combining Cubes. (default: +)
  --new-name=STR             New name for the resulting dataset.
  --new-metadata=INPUTJSON   Dictionary of required changes to the metadata.  Default is None.
  --warnings-on              If True, warning messages where metadata do not match will be given.  Default is False.
  --output=STR               Output file name

Other actions:
  -h, --help                 Show the help

Returns a cube with the combined data.
__HELP__
  [[ "$output" == "$expected" ]]
}
