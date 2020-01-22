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

TIGHT_TOLERANCE = 1e-5
DEFAULT_TOLERANCE = 1e-4
LOOSE_TOLERANCE = 1e-3


def compare_netcdfs(actual_path, desired_path, rtol, atol,
                    exclude_vars=None, reporter=None):
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
    except OSError as e:
        reporter(str(e))
        return
    try:
        desired_ds = netCDF4.Dataset(str(desired_path), mode='r')
    except OSError as e:
        reporter(str(e))
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
    actual_groups = sorted(actual_ds.groups.keys())
    desired_groups = sorted(desired_ds.groups.keys())

    if actual_groups != desired_groups:
        msg = f"different groups {name}: {actual_groups} {desired_groups}"
        reporter(msg)

    for group in desired_groups:
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
    actual_dims = sorted(set(actual_ds.dimensions.keys()) - set(exclude_vars))
    desired_dims = sorted(
        set(desired_ds.dimensions.keys()) - set(exclude_vars))

    if actual_dims != desired_dims:
        msg = ("different dimensions - "
               f"{name} {actual_dims} {desired_dims}")
        reporter(msg)

    for dim in desired_dims:
        try:
            actual_len = actual_ds.dimensions[dim].size
            desired_len = desired_ds.dimensions[dim].size
            if actual_len != desired_len:
                msg = ("different dimension size - "
                       f"{name}/{dim} {actual_len} {desired_len}")
                reporter(msg)
        except KeyError:
            pass


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
    actual_vars = sorted(set(actual_ds.variables.keys()) - set(exclude_vars))
    desired_vars = sorted(set(desired_ds.variables.keys()) - set(exclude_vars))

    if actual_vars != desired_vars:
        msg = ("different variables - "
               f"{name} {actual_vars} {desired_vars}")

        reporter(msg)

    coords = set()
    for var in desired_vars:
        coords = coords.union(desired_ds.variables[var].dimensions)
    metadata_vars = coords.intersection(desired_vars)

    for var in desired_vars:
        var_path = f"{name}/{var}"
        try:
            actual_var = actual_ds.variables[var]
            desired_var = desired_ds.variables[var]
        except KeyError:
            continue
        compare_attributes(var_path, actual_var, desired_var, reporter)
        if var in exclude_vars:
            pass
        elif var in metadata_vars:
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
    actual_attrs = sorted(actual_ds.ncattrs())
    desired_attrs = sorted(desired_ds.ncattrs())
    # ignore history attribute - this often contain datestamps and other
    # overly specific details
    if "history" in actual_attrs:
        actual_attrs.remove("history")
    if "history" in desired_attrs:
        desired_attrs.remove("history")

    if actual_attrs != desired_attrs:
        msg = (f"different attributes of {name} -"
               f" {actual_attrs} {desired_attrs}")
        reporter(msg)

    for key in desired_attrs:
        try:
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
        except KeyError:
            pass


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
        msg = f"different type {name} - {actual_var.type} {desired_var.type}"
        reporter(msg)
    actual_data = actual_var[:]
    desired_data = desired_var[:]
    difference_found = False
    numpy_err_message = ''
    try:
        if actual_data.dtype.kind in ['b', 'O', 'S', 'U', 'V']:
            # numpy boolean, object, bytestring, unicode and void types don't
            # have numerical "closeneess" so use exact equality for these
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
