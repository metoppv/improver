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
"""IMPROVER version."""

# adapted from CC0 Public Domain Dedication code:
# https://github.com/Changaco/version.py

import re

ABREV_HASH = '$Format:%h$'
FULL_HASH = '$Format:%H$'
REF_NAMES = '$Format:%D$'

tag_re = re.compile(r'\btag: ([0-9][^,]*)\b')
version_re = re.compile('^Version: (.+)$', re.M)


def get_version():
    """Return version based on git tag.
    """
    import os
    from subprocess import CalledProcessError, check_output
    version = tag_re.search(REF_NAMES)
    if version:
        return version.group(1)

    d = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if os.path.isdir(os.path.join(d, '.git')):
        # Get the version using "git describe".
        cmd = 'git describe --tags --match [0-9]* --dirty'
        cwd = os.getcwd()
        try:
            os.chdir(d)
            version = check_output(cmd.split()).decode().strip()
        except CalledProcessError:
            raise RuntimeError('Unable to get version number from git tags')
        finally:
            os.chdir(cwd)

        # PEP 440 compatibility
        if '-' in version:
            dirty = version.endswith('-dirty')
            version = '.post'.join(version.split('-')[:2])
            if dirty:
                version += '.dirty'

    else:
        # Extract the version from the PKG-INFO file.
        try:
            with open(os.path.join(d, 'PKG-INFO')) as f:
                version = version_re.search(f.read()).group(1)
        except FileNotFoundError:
            version = 'unknown'

    return version
