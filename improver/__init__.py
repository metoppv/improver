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
"""Module containing plugin base class."""
from abc import ABC, abstractmethod


class BasePlugin(ABC):
    """An abstract class for IMPROVER plugins.
    Subclasses must be callable. We preserve the process
    method by redirecting to __call__.
    """
    def __call__(self, *args, **kwargs):
        """Makes subclasses callable to use process
        Args:
            *args:
                Positional arguments.
            **kwargs:
                Keyword arguments.
        Returns:
            Output of self.process()
        """
        return self.process(*args, **kwargs)

    @abstractmethod
    def process(self, *args, **kwargs):
        """Abstract class for rest to implement."""
        pass


class PostProcessingPlugin(BasePlugin):
    """An abstract class for IMPROVER post-processing plugins.
    Makes generalised changes to metadata relating to post-processing.
    """
    def __call__(self, *args, **kwargs):
        """Makes subclasses callable to use process
        Args:
            *args:
                Positional arguments.
            **kwargs:
                Keyword arguments.
        Returns:
            iris.cube.Cube:
                Output of self.process() with updated title attribute
        """
        cube = super().__call__(*args, **kwargs)
        self.post_processed_title(cube)
        return cube

    @staticmethod
    def post_processed_title(cube):
        """Updates title attribute on output cube to include
        "Post-Processed"
        """
        from improver.metadata.constants.attributes import \
            MANDATORY_ATTRIBUTE_DEFAULTS
        default_title = MANDATORY_ATTRIBUTE_DEFAULTS["title"]
        if ("title" in cube.attributes.keys() and
                cube.attributes["title"] != default_title and
                "Post-Processed" not in cube.attributes["title"]):
            title = cube.attributes["title"]
            cube.attributes["title"] = f"Post-Processed {title}"
