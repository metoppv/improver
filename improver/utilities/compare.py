#!/usr/bin/env python
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
"""
Compare netcdf files using python-netCDF4 library.

This comparison is done using netCDF4 rather than iris so that it is not
coupled to changes in netCDF representation across iris versions. It is also
able to compare non-CF conventions compliant netCDF files that iris has
difficulty loading.

Many functions in this module take an argument called 'reporter' which is a
function to be called to report comparison differences. This provides
flexibility regarding what action should be taken when differences are found.
For example, the action to take could be to print the message, log the message,
or raise an appropriate exception.
"""

import netCDF4
import numpy as np

from improver.constants import DEFAULT_TOLERANCE


def compare_netcdfs(actual_path, desired_path, rtol=DEFAULT_TOLERANCE,
                    atol=DEFAULT_TOLERANCE, exclude_vars=None, reporter=None):
    """
    Compare two netCDF files.

    Args:
        actual_path (os.Pathlike): data file produced by test run
        desired_path (os.Pathlike): data file considered good eg. KGO
        rtol (float): relative tolerance
        atol (float): absolute tolerance
        exclude_vars (Iterable[str]): variable names to exclude from comparison
        reporter (Callable[[str], None]): callback function for
            reporting differences

    Returns:
        None
    """
    def raise_reporter(message):
        raise ValueError(message)
    if exclude_vars is None:
        exclude_vars = []
    if reporter is None:
        reporter = raise_reporter

    try:
        actual_ds = netCDF4.Dataset(str(actual_path), mode='r')
    except OSError as exc:
        reporter(str(exc))
        return
    try:
        desired_ds = netCDF4.Dataset(str(desired_path), mode='r')
    except OSError as exc:
        reporter(str(exc))
        return
    desired_ds.set_auto_maskandscale(False)
    actual_ds.set_auto_maskandscale(False)
    compare_datasets("", actual_ds, desired_ds, rtol, atol,
                     exclude_vars, reporter)


def compare_datasets(name, actual_ds, desired_ds, rtol, atol,
                     exclude_vars, reporter):
    """
    Compare netCDF datasets.
    This function can call itself recursively to handle nested groups in
    netCDF4 files which are represented using the same Dataset class by
    python-netCDF4.

    Args:
        name (str): group name
        actual_ds (netCDF.Dataset): dataset produced by test run
        desired_ds (netCDF.Dataset): dataset considered good
        rtol (float): relative tolerance
        atol (float): absolute tolerance
        exclude_vars (List[str]): variable names to exclude from comparison
        reporter (Callable[[str], None]): callback function for
            reporting differences

    Returns:
        None
    """
    compare_attributes("root", actual_ds, desired_ds, reporter)
    actual_groups = set(actual_ds.groups.keys())
    desired_groups = set(desired_ds.groups.keys())

    if actual_groups != desired_groups:
        msg = (f"different groups {name}: "
               f"{sorted(actual_groups ^ desired_groups)}")
        reporter(msg)

    check_groups = actual_groups.intersection(desired_groups)
    for group in check_groups:
        compare_attributes(group, actual_ds.groups[group],
                           desired_ds.groups[group], reporter)
        compare_datasets(group,
                         actual_ds.groups[group], desired_ds.groups[group],
                         rtol, atol, exclude_vars, reporter)

    compare_dims(name, actual_ds, desired_ds, exclude_vars, reporter)
    compare_vars(name, actual_ds, desired_ds, rtol, atol,
                 exclude_vars, reporter)


def compare_dims(name, actual_ds, desired_ds, exclude_vars, reporter):
    """
    Compare dimensions in a netCDF dataset/group.

    Args:
        name (str): group name
        actual_ds (netCDF.Dataset): dataset produced by test run
        desired_ds (netCDF.Dataset): dataset considered good
        exclude_vars (List[str]): variable names to exclude from comparison
        reporter (Callable[[str], None]): callback function for
            reporting differences

    Returns:
        None
    """
    if exclude_vars is None:
        exclude_vars = []
    actual_dims = set(actual_ds.dimensions.keys()) - set(exclude_vars)
    desired_dims = set(desired_ds.dimensions.keys()) - set(exclude_vars)

    if actual_dims != desired_dims:
        msg = ("different dimensions - "
               f"{name} {sorted(actual_dims ^ desired_dims)}")
        reporter(msg)

    check_dims = actual_dims.intersection(desired_dims)
    for dim in check_dims:
        actual_len = actual_ds.dimensions[dim].size
        desired_len = desired_ds.dimensions[dim].size
        if actual_len != desired_len:
            msg = ("different dimension size - "
                   f"{name}/{dim} {actual_len} {desired_len}")
            reporter(msg)


def compare_vars(name, actual_ds, desired_ds, rtol, atol,
                 exclude_vars, reporter):
    """
    Compare variables in a netCDF dataset/group.

    Args:
        name (str): group name
        actual_ds (netCDF.Dataset): dataset produced by test run
        desired_ds (netCDF.Dataset): dataset considered good
        rtol (float): relative tolerance
        atol (float): absolute tolerance
        exclude_vars (List[str]): variable names to exclude from comparison
        reporter (Callable[[str], None]): callback function for
            reporting differences

    Returns:
        None
    """
    if exclude_vars is None:
        exclude_vars = []
    actual_vars = set(actual_ds.variables) - set(exclude_vars)
    desired_vars = set(desired_ds.variables) - set(exclude_vars)

    if actual_vars != desired_vars:
        msg = (f"different variables - {name} "
               f"{sorted(actual_vars ^ desired_vars)}")
        reporter(msg)

    check_vars = actual_vars.intersection(desired_vars)
    # coordinate variables are those used as dimensions on other variables
    # these should match exactly to avoid mis-alignment issues
    coord_vars = set()
    for var in check_vars:
        coord_vars = coord_vars.union(set(desired_ds[var].dimensions))
    coord_vars = coord_vars.intersection(check_vars)

    for var in check_vars:
        var_path = f"{name}/{var}"
        actual_var = actual_ds.variables[var]
        desired_var = desired_ds.variables[var]
        compare_attributes(var_path, actual_var, desired_var, reporter)
        if var in coord_vars:
            compare_data(var_path, actual_var, desired_var,
                         0.0, 0.0, reporter)
        else:
            compare_data(var_path, actual_var, desired_var,
                         rtol, atol, reporter)


def compare_attributes(name, actual_ds, desired_ds, reporter):
    """
    Compare attributes in a netCDF dataset/group.

    Args:
        name (str): group name
        actual_ds (netCDF.Dataset): dataset produced by test run
        desired_ds (netCDF.Dataset): dataset considered good
        reporter (Callable[[str], None]): callback function for
            reporting differences

    Returns:
        None
    """
    actual_attrs = set(actual_ds.ncattrs())
    desired_attrs = set(desired_ds.ncattrs())
    # ignore history attribute - this often contain datestamps and other
    # overly specific details
    actual_attrs.discard("history")
    desired_attrs.discard("history")

    if actual_attrs != desired_attrs:
        msg = (f"different attributes of {name} - "
               f"{sorted(actual_attrs ^ desired_attrs)}")
        reporter(msg)

    check_attrs = actual_attrs.intersection(desired_attrs)
    for key in check_attrs:
        actual_attr = actual_ds.getncattr(key)
        desired_attr = desired_ds.getncattr(key)
        assert isinstance(desired_attr, type(actual_attr))
        if isinstance(desired_attr, np.ndarray):
            if not np.array_equal(actual_attr, desired_attr):
                msg = (f"different attribute value {name}/{key} - "
                       f"{actual_attr} {desired_attr}")
                reporter(msg)
        elif actual_attr != desired_attr:
            msg = (f"different attribute value {name}/{key} - "
                   f"{actual_attr} {desired_attr}")
            reporter(msg)


def compare_data(name, actual_var, desired_var, rtol, atol, reporter):
    """
    Compare attributes in a netCDF variable.

    Args:
        name (str): variable name
        actual_var (netCDF.Variable): variable produced by test run
        desired_var (netCDF.Variable): variable considered good
        rtol (float): relative tolerance
        atol (float): absolute tolerance
        reporter (Callable[[str], None]): callback function for
            reporting differences

    Returns:
        None
    """
    if actual_var.dtype != desired_var.dtype:
        msg = f"different type {name} - {actual_var.dtype} {desired_var.dtype}"
        reporter(msg)
    actual_data = actual_var[:]
    desired_data = desired_var[:]
    difference_found = False
    numpy_err_message = ''
    try:
        if actual_data.dtype.kind in ['b', 'O', 'S', 'U', 'V']:
            # numpy boolean, object, bytestring, unicode and void types don't
            # have numerical "closeness" so use exact equality for these
            np.testing.assert_equal(actual_data, desired_data, verbose=True)
        else:
            np.testing.assert_allclose(actual_data, desired_data, rtol, atol,
                                       equal_nan=True, verbose=True)
    except AssertionError as exc:
        difference_found = True
        numpy_err_message = str(exc).strip()
    # call the reporter function outside the except block to avoid nested
    # exceptions if the reporter function is raising an exception
    if difference_found:
        reporter(f"different data {name} - {numpy_err_message}")
