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
"""Utilities for Unit tests for Weather Symbols"""


def check_diagnostic_lists_consistency(query):
    """
    Checks if specific input lists have same nested list
    structure. e.g. ['item'] != [['item']]

    Args:
        query (dict):
            of weather-symbols decision-making information

    Raises:
        ValueError: if diagnostic query lists have different nested list
            structure.

    """
    diagnostic_keys = [
        'diagnostic_fields',
        'diagnostic_conditions',
        'diagnostic_thresholds']
    values = [query[key] for key in diagnostic_keys]
    if not check_nested_list_consistency(values):
        msg = f"Inconsistent list structure: \n"
        for key in diagnostic_keys:
            msg += f"{key} = {query[key]}; \n"
        raise ValueError(msg)


def check_nested_list_consistency(query):
    """
    Return True if all input lists have same nested list
    structure. e.g. ['item'] != [['item']]

    Args:
        query (list of lists):
            Nested lists to check for consistency.

    Returns:
        bool: True if diagnostic query lists have same nested list
            structure.

    """

    def _checker(lists):
        """Return True if all input lists have same nested list
        structure. e.g. ['item'] != [['item']]."""
        type_set = set(map(type, lists))
        if list in type_set:
            return (
                    len(type_set) == 1 and
                    len(set(map(len, lists))) == 1 and
                    all(map(_checker, zip(*lists)))
            )
        return True

    return _checker(query)
