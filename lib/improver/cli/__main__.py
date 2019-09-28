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
"""Main to run clize on the cli"""
from improver import cli
import pkgutil


# help command

@cli.clizefy(
    with_output=False,
    help_names=(),  # no help --help
)
def improver_help(progname: cli.pass_program_name,
                  command=None, *, usage=False):
    """Show command help."""
    progname = progname.partition(' ')[0]
    args = [command, '--help', usage and '--usage']
    return cli.execute_command(SUBCOMMANDS_CALLABLE, progname,
                               *filter(None, args))


# mapping of command names to CLI objects

SUBCOMMANDS_TABLE = {
    'help': improver_help,
}
for m in pkgutil.iter_modules(cli.__path__):
    if m.name != '__main__':
        climod = __import__('improver.cli.' + m.name, fromlist=['process'])
        SUBCOMMANDS_TABLE[m.name] = cli.get_cli(climod.process)

# main CLI object with subcommands

SUBCOMMANDS_CALLABLE = cli.get_cli(
    SUBCOMMANDS_TABLE,
    description="""IMPROVER NWP post-processing toolbox""",
    footnotes="""See also improver --help for more information.""")


# IMPROVER main


@cli.clizefy(with_output=False)
def main(progname: cli.pass_program_name,
         command: cli.Parameter.LAST_OPTION,
         *args,
         verbose=False,
         dry_run=False):
    """IMPROVER NWP post-processing toolbox

    Results from commands can be passed into file-like arguments
    of other commands by surrounding them by square brackets::

        improver command [ command ... ] ...

    Spaces around the brackets are mandatory.

    Args:
        command (str):
            Command to execute
        args (tuple):
            Command arguments
        verbose (bool):
            Print executed commands
        dry_run (bool):
            Print commands to be executed

    See ``improver help [--usage] [command]`` for more information
    on available command(s).
    """
    return cli.execute_command(SUBCOMMANDS_CALLABLE,
                               progname, command, *args[:],
                               verbose=verbose, dry_run=dry_run)


if __name__ == '__main__':
    cli.run(main)
