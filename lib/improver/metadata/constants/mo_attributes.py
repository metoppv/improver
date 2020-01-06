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
"""Module defining Met Office specific attributes"""

GRID_TYPE = 'standard'
STAGE_VERSION = '1.3.0'

MOSG_GRID_ATTRIBUTES = {
    'mosg__grid_type', 'mosg__grid_version', 'mosg__grid_domain'}

# Define current StaGE grid metadata
MOSG_GRID_DEFINITION = {
    'uk_ens': {'mosg__grid_type': GRID_TYPE,
               'mosg__model_configuration': 'uk_ens',
               'mosg__grid_domain': 'uk_extended',
               'mosg__grid_version': STAGE_VERSION},
    'gl_ens': {'mosg__grid_type': GRID_TYPE,
               'mosg__model_configuration': 'gl_ens',
               'mosg__grid_domain': 'global',
               'mosg__grid_version': STAGE_VERSION},
    'uk_det': {'mosg__grid_type': GRID_TYPE,
               'mosg__model_configuration': 'uk_det',
               'mosg__grid_domain': 'uk_extended',
               'mosg__grid_version': STAGE_VERSION},
    'gl_det': {'mosg__grid_type': GRID_TYPE,
               'mosg__model_configuration': 'gl_det',
               'mosg__grid_domain': 'global',
               'mosg__grid_version': STAGE_VERSION}
}

# Map correct metadata from StaGE v1.1.0
GRID_ID_LOOKUP = {'enukx_standard_v1': 'uk_ens',
                  'engl_standard_v1': 'gl_ens',
                  'ukvx_standard_v1': 'uk_det',
                  'glm_standard_v1': 'gl_det'}
