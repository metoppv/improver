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

@test "tests bad option" {
  run improver tests --silly-option
  [[ "$status" -eq 2 ]]
  read -d '' expected <<'__HELP__' || true
improver tests [OPTIONS] [SUBTEST...]

Run pycodestyle, pylint, documentation, unit and CLI acceptance tests.

Optional arguments:
    --bats          Run CLI tests using BATS instead of the default prove
    --debug         Run in verbose mode (may take longer for CLI)
    -h, --help      Show this message and exit

Arguments:
    SUBTEST         Name(s) of a subtest to run without running the rest.
                    Valid names are: pycodestyle, pylint, pylintE, licence, doc, unit, cli.
                    pycodestyle, pylintE, licence, doc, unit, and cli are the default tests.
    SUBCLI          Name(s) of cli subtests to run without running the rest.
                    Valid names are tasks which appear in /improver/tests/
                    without the "improver-" prefix. The default is to run all
                    cli tests in the /improver/tests/ directory.
                    e.g. 'improver tests cli nbhood' will run neighbourhood
                    processing cli tests only.
__HELP__
  [[ "$output" == "$expected" ]]
}
