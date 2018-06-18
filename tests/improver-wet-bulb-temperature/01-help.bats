#!/usr/bin/env bats
# -----------------------------------------------------------------------------
# (C) British Crown Copyright 2017-2018 Met Office.
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

@test "wet-bulb-temperature -h" {
  run improver wet-bulb-temperature -h
  [[ "$status" -eq 0 ]]
  read -d '' expected <<'__HELP__' || true
usage: improver-wet-bulb-temperature [-h] [--profile]
                                     [--profile_file PROFILE_FILE]
                                     [--convergence_condition CONVERGENCE_CONDITION]
                                     TEMPERATURE RELATIVE_HUMIDITY PRESSURE
                                     OUTPUT_FILE

Calculate a field of wet bulb temperatures.

positional arguments:
  TEMPERATURE           Path to a NetCDF file of air temperatures at the
                        points for which the wet bulb temperatures are being
                        calculated.
  RELATIVE_HUMIDITY     Path to a NetCDF file of relative humidities at the
                        points for for which the wet bulb temperatures are
                        being calculated.
  PRESSURE              Path to a NetCDF file of air pressures at the points
                        for which the wet bulb temperatures are being
                        calculated.
  OUTPUT_FILE           The output path for the processed NetCDF.

optional arguments:
  -h, --help            show this help message and exit
  --profile             Switch on profiling information.
  --profile_file PROFILE_FILE
                        Dump profiling info to a file. Implies --profile.
  --convergence_condition CONVERGENCE_CONDITION
                        The convergence condition for the Newton iterator in
                        K. When the wet bulb temperature stops changing by
                        more than this amount between iterations, the solution
                        is accepted.
__HELP__
  [[ "$output" == "$expected" ]]
}
