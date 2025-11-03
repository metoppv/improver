# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing plugin base class."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version
from typing import Any

try:
    __version__ = version("improver")
except PackageNotFoundError:
    # package is not installed
    pass


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
    def process(self, *args, **kwargs) -> Any:
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
            Output of self.process() with updated title attribute
        """
        from iris.cube import Cube

        result = super().__call__(*args, **kwargs)
        if isinstance(result, Cube):
            self.post_processed_title(result)
        elif isinstance(result, Iterable) and not isinstance(result, str):
            for item in result:
                if isinstance(item, Cube):
                    self.post_processed_title(item)
        return result

    @staticmethod
    def post_processed_title(cube):
        """Updates title attribute on output cube to include
        "Post-Processed"
        """
        from improver.metadata.constants.attributes import MANDATORY_ATTRIBUTE_DEFAULTS

        default_title = MANDATORY_ATTRIBUTE_DEFAULTS["title"]
        if (
            "title" in cube.attributes.keys()
            and cube.attributes["title"] != default_title
            and "Post-Processed" not in cube.attributes["title"]
        ):
            title = cube.attributes["title"]
            cube.attributes["title"] = f"Post-Processed {title}"
