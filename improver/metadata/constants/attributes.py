# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Defaults for mandatory attributes"""

MANDATORY_ATTRIBUTE_DEFAULTS = {
    "title": "unknown",
    "source": "IMPROVER",
    "institution": "unknown",
}

MANDATORY_ATTRIBUTES = [x for x in MANDATORY_ATTRIBUTE_DEFAULTS.keys()]
