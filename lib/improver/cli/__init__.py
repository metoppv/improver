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
"init for cli and clize"
import clize

from improver.cli.clize_routines import clizefy
from improver.cli.combine import combine_process


def preprocess_command(progname, command, *args):
    """Preprocess command before execution."""
    post_args = []
    outopt = '--output'
    for i, arg in enumerate(args, 1):
        if isinstance(arg, str) and arg.startswith(outopt):
            _, sep, output = arg.partition('=')
            if not sep:
                try:
                    output = args[i + 1]
                except IndexError:
                    output = ''
            post_args.extend([outopt, output])
            break
    return command, args, post_args


# alias to reuse in replacement later on
prep_cmd = preprocess_command


# helper function to execute the main CLI object
# (relaced later on to achieve subcommand chaining)
def process_command(progname, command, *args, verbose=False):
    """Common entry point for command execution."""
    result = CLI(progname, command, *args)
    if verbose:
        rtype = type(result)
        output = '%s.%s@%i' % (rtype.__module__, rtype.__name__, id(result))
        print(progname, command, *args, ' -> ', output)
    return result


# suppress output if saved already
def postprocess_command(progname, result, *post_args):
    """Postprocess result from command execution."""
    try:
        output = post_args[post_args.index('--output') + 1]
    except (ValueError, IndexError):
        return result  # we could reraise here to require output
    return


# IMPROVER main

@clizefy(with_output=False)
def main(progname: clize.parameters.pass_name,
         command: clize.Parameter.LAST_OPTION,
         *args,
         verbose: 'v' = False,
         command_preprocessor: clize.Parameter.IGNORE = None,
         command_processor: clize.Parameter.IGNORE = None,
         command_postprocessor: clize.Parameter.IGNORE = None):
    """IMPROVER post-processing toolbox

    Args:
        command (str):
            Command to execute
        args (tuple):
            Command arguments
        verbose (bool):
            Print executed commands

    See ``improver help [--usage] [command]`` for more information
    on available command(s).
    """
    command_preprocessor = command_preprocessor or preprocess_command
    command_processor = command_processor or process_command
    command_postprocessor = command_postprocessor or postprocess_command

    command, args, post_args = command_preprocessor(progname, command, *args)
    result = command_processor(progname, command, *args, verbose=verbose)
    return command_postprocessor(progname, result, *post_args)


# help command

@clizefy(
    with_output=False,
    help_names=(),  # no help --help
)
def improver_help(progname: clize.parameters.pass_name,
                  command=None, *, usage=False):
    """Show command help."""
    progname = progname.partition(' ')[0]
    args = [command, '--help', usage and '--usage']
    return process_command(progname, *filter(None, args))


# mapping of command names to CLI objects
# 'combine': combine_process = improver.cli.combine.process

cli_table = {
    'help': improver_help,
    'combine': combine_process
}

# main CLI object with subcommands

CLI = clize.Clize.get_cli(
    cli_table, description=main.cli.helper.description,
    footnotes="""See also improver --help for more information.""")
