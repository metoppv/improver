# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

"""module to give access to callable methods on objects."""


def call_object_method(obj: object, method_name: str, **kwargs):
    """
    Calls a method on an object with the supplied arguments.

    This method allows us to construct a callable method for DAGRunner to execute
    where the method to be called is on an object that comes from another plugin.

    e.g. cube.collapsed("height", iris.analysis.SUM) becomes
         call_object_method(cube, "collapsed", coords="height", aggregator=iris.analysis.SUM)

    Args:
        obj:
            The object containing the method to be called.
        method_name:
            The name of the method to be called.
        **kwargs:
            The keyword arguments to be passed to the method.

    Returns:
        The return value from the called method.
    """
    method = getattr(obj, method_name)
    return method(**kwargs)
