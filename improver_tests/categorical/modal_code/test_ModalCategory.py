# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for ModalCategory class."""

from calendar import timegm
from datetime import datetime as dt
from datetime import timedelta
from typing import Tuple

import numpy as np
import pytest
from iris.cube import CubeList

from improver.blending import WEIGHT_FORMAT
from improver.categorical.modal_code import ModalCategory
from improver.metadata.constants.time_types import DT_FORMAT
from improver.spotdata.build_spotdata_cube import build_spotdata_cube
from improver.synthetic_data.set_up_test_cubes import construct_scalar_time_coords
from improver_tests.categorical.decision_tree import set_up_wxcube, wxcode_decision_tree

MODEL_ID_ATTR = "mosg__model_configuration"
RECORD_RUN_ATTR = "mosg__model_run"
TARGET_TIME = dt(2020, 6, 15, 18)


@pytest.fixture(name="wxcode_series")
def wxcode_series_fixture(
    data,
    cube_type,
    interval: int,
    offset_reference_times: bool,
    model_id_attr: bool,
    record_run_attr: bool,
) -> Tuple[bool, CubeList]:
    """Generate a time series of weather code cubes for combination to create
    a period representative code. When offset_reference_times is set, each
    successive cube will have a reference time one hour older."""

    time = TARGET_TIME

    data = np.array(data)
    if len(data.shape) > 1:
        data = data.T

    ntimes = len(data)
    wxcubes = CubeList()

    for i in range(ntimes):
        wxtime = time - timedelta(hours=i * interval)
        wxbounds = [wxtime - timedelta(hours=interval), wxtime]
        if offset_reference_times:
            wxfrt = time - timedelta(hours=42) - timedelta(hours=i)
        else:
            wxfrt = time - timedelta(hours=42)
        wxdata = np.ones((2, 2), dtype=np.int8)

        if len(data[i].shape) > 0 and np.product(wxdata.shape) == data[i].shape[0]:
            wxdata = np.reshape(data[i], wxdata.shape)
        else:
            if len(data[i].shape) == 0:
                index = 0
            else:
                index = slice(None, len(data[i]))
            wxdata[0, index] = data[i]

        if cube_type == "gridded":
            wxcubes.append(
                set_up_wxcube(data=wxdata, time=wxtime, time_bounds=wxbounds, frt=wxfrt)
            )
        else:
            time_coords = construct_scalar_time_coords(wxtime, wxbounds, wxfrt)
            time_coords = [crd for crd, _ in time_coords]
            latitudes = np.array([50, 52, 54, 56])
            longitudes = np.array([-4, -2, 0, 2])
            altitudes = wmo_ids = unique_site_id = np.arange(4)
            unique_site_id_key = "met_office_site_id"
            wxcubes.append(
                build_spotdata_cube(
                    wxdata.flatten(),
                    "weather_code",
                    1,
                    altitudes,
                    latitudes,
                    longitudes,
                    wmo_ids,
                    unique_site_id=unique_site_id,
                    unique_site_id_key=unique_site_id_key,
                    scalar_coords=time_coords,
                )
            )

        # Add a blendtime coordinate as UK weather symbols are constructed
        # from model blended data.
        blend_time = wxcubes[-1].coord("forecast_reference_time").copy()
        blend_time.rename("blend_time")
        wxcubes[-1].add_aux_coord(blend_time)

        if model_id_attr:
            if i == 0:
                wxcubes[-1].attributes.update({MODEL_ID_ATTR: "uk_det uk_ens"})
            else:
                wxcubes[-1].attributes.update({MODEL_ID_ATTR: "uk_ens"})

        if record_run_attr and model_id_attr:
            ukv_time = wxfrt - timedelta(hours=1)
            enukx_time = wxfrt - timedelta(hours=3)
            if i == 0:
                ukv_weight = 0.5
                enukx_weight = 0.5
                wxcubes[-1].attributes.update(
                    {
                        RECORD_RUN_ATTR: (
                            f"uk_det:{ukv_time:{DT_FORMAT}}:{ukv_weight:{WEIGHT_FORMAT}}\n"
                            f"uk_ens:{enukx_time:{DT_FORMAT}}:{enukx_weight:{WEIGHT_FORMAT}}"
                        )
                    }
                )
            else:
                enukx_weight = 1.0
                wxcubes[-1].attributes.update(
                    {
                        RECORD_RUN_ATTR: (
                            f"uk_ens:{enukx_time:{DT_FORMAT}}:{enukx_weight:{WEIGHT_FORMAT}}"
                        )
                    }
                )

    return interval, model_id_attr, record_run_attr, offset_reference_times, wxcubes


@pytest.mark.parametrize("record_run_attr", [False])
@pytest.mark.parametrize("model_id_attr", [False, True])
@pytest.mark.parametrize("interval", [1])
@pytest.mark.parametrize("offset_reference_times", [False, True])
@pytest.mark.parametrize("cube_type", ["gridded", "spot"])
@pytest.mark.parametrize(
    "data, expected",
    (
        # Sunny day (1), one rain code (15) that is in the minority, expect sun
        # code (1).
        ([1, 1, 1, 15], 1),
        # Short period with an equal split. The most significant weather
        # (hail, 21) should be returned.
        ([1, 21], 21),
        # A single time is provided in which a sleet shower is forecast (16).
        # We expect the cube to be returned with the night code changed to a
        # day code (17).
        ([16], 17),
        # Equal split in day codes, but a night code corresponding to one
        # of the day types means a valid mode can be calculated. We expect the
        # day code (10) to be returned.
        ([1, 1, 10, 10, 9], 10),
        # No clear representative code. Falls back to grouping, which
        # consolidates the codes containing rain (10, 11, 12, 14, 15) and yields
        # the least significant of these that is present (10).
        ([1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 14, 15], 10),
        # No clear representative code. Falls back to grouping, which
        # consolidates the codes containing rain (10, 11, 12, 14, 15) and yields
        # the least significant of these which is present (11); the light
        # shower code (10) is not present, so will not be picked.
        ([1, 3, 4, 5, 6, 7, 8, 16, 11, 12, 14, 15], 11),
        # No clear representative code. This falls back to grouping,
        # consolidates the codes containing visibility (5, 6) and yields
        # the least significant of these which is present (5).
        ([5, 5, 5, 5, 6, 6, 6, 6, 8, 8, 8, 8, 7, 7, 7, 7], 5),
        # An extreme edge case in which all the codes across time for a site
        # are different. All the codes fall into different groups and cannot be
        # consolidated. In this case the most significant weather from the whole
        # set is returned. In this case that is a light snow shower (23).
        ([1, 3, 4, 5, 7, 8, 10, 17, 20, 23], 23),
    ),
)
def test_expected_values(wxcode_series, expected):
    """Test that the expected period representative symbol is returned."""
    _, _, _, _, wxcode_cubes = wxcode_series
    result = ModalCategory(wxcode_decision_tree())(wxcode_cubes)
    assert result.data.flatten()[0] == expected


@pytest.mark.parametrize("record_run_attr", [False, True])
@pytest.mark.parametrize("model_id_attr", [False, True])
@pytest.mark.parametrize("interval", [1, 3])
@pytest.mark.parametrize("offset_reference_times", [False, True])
@pytest.mark.parametrize("cube_type", ["gridded", "spot"])
@pytest.mark.parametrize("data", [np.ones(12), np.ones(1)])
def test_metadata(wxcode_series):
    """Check that the returned metadata is correct. In this case we expect a
    time coordinate with bounds that describe the full period over which the
    representative symbol has been calculated while the forecast_reference_time
    will be the latest of those input and the forecast_period will be the
    difference between the forecast_reference_time and time.

    A single data point is tested which means a single cube is passed in. This
    ensures the metadata is consistent whether or not the input data has passed
    through the modal aggregator."""

    def as_utc_timestamp(time):
        return timegm(time.utctimetuple())

    (
        interval,
        model_id_attr,
        record_run_attr,
        offset_reference_times,
        wxcode_cubes,
    ) = wxcode_series

    kwargs = {}
    if model_id_attr:
        kwargs.update({"model_id_attr": MODEL_ID_ATTR})
    if record_run_attr:
        kwargs.update({"record_run_attr": RECORD_RUN_ATTR})

    result = ModalCategory(wxcode_decision_tree(), **kwargs)(wxcode_cubes)

    n_times = len(wxcode_cubes)
    expected_time = TARGET_TIME
    expected_bounds = [TARGET_TIME - timedelta(hours=n_times * interval), TARGET_TIME]
    expected_reference_time = TARGET_TIME - timedelta(hours=42)
    expected_forecast_period = (expected_time - expected_reference_time).total_seconds()
    expected_forecast_period_bounds = [
        expected_forecast_period - n_times * interval * 3600,
        expected_forecast_period,
    ]
    expected_model_id_attr = "uk_det uk_ens"
    expected_record_det = "uk_det:20200613T2300Z:\n"
    expected_record_ens = "uk_ens:20200613T{}00Z:"

    # Expected record_run attribute contains all contributing cycle times.
    if offset_reference_times and len(wxcode_cubes) > 1:
        expected_record_run_attr = expected_record_det + "\n".join(
            [expected_record_ens.format(value) for value in range(10, 22)]
        )
    else:
        expected_record_run_attr = expected_record_det + expected_record_ens.format(21)

    assert result.coord("time").points[0] == as_utc_timestamp(expected_time)
    assert result.coord("time").bounds[0][0] == as_utc_timestamp(expected_bounds[0])
    assert result.coord("time").bounds[0][1] == as_utc_timestamp(expected_bounds[1])
    assert result.coord("forecast_reference_time").points[0] == as_utc_timestamp(
        expected_reference_time
    )
    assert not result.coord("forecast_reference_time").has_bounds()
    assert result.coord("forecast_period").points[0] == expected_forecast_period
    assert np.allclose(
        result.coord("forecast_period").bounds[0], expected_forecast_period_bounds
    )
    assert result.cell_methods[0].method == "mode"
    assert result.cell_methods[0].coord_names[0] == "time"
    assert result.cell_methods[0].intervals[0] == f"{interval} hour"
    if model_id_attr:
        assert result.attributes[MODEL_ID_ATTR] == expected_model_id_attr
    else:
        assert MODEL_ID_ATTR not in result.attributes.keys()
    if record_run_attr and model_id_attr:
        assert RECORD_RUN_ATTR in result.attributes.keys()
        assert result.attributes[RECORD_RUN_ATTR] == expected_record_run_attr
    else:
        assert RECORD_RUN_ATTR not in result.attributes.keys()


@pytest.mark.parametrize("record_run_attr", [False])
@pytest.mark.parametrize("model_id_attr", [False])
@pytest.mark.parametrize("interval", [1, 3])
@pytest.mark.parametrize("offset_reference_times", [False])
@pytest.mark.parametrize("cube_type", ["gridded", "spot"])
@pytest.mark.parametrize("data", [[1, 1, 1, 15]])
def test_unmatching_bounds_exception(wxcode_series):
    """Test that an exception is raised if inputs do not represent the same
    intervals."""
    _, _, _, _, wxcode_cubes = wxcode_series
    bounds = wxcode_cubes[0].coord("time").bounds.copy()
    bounds[0][0] += 1800
    wxcode_cubes[0].coord("time").bounds = bounds
    with pytest.raises(
        ValueError, match="Input diagnostics do not have consistent periods."
    ):
        ModalCategory(wxcode_decision_tree())(wxcode_cubes)
