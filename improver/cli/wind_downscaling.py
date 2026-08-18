#!/usr/bin/env python
# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to apply unresolved-orography wind corrections."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    target_wind_speed_cube: cli.inputcube,
    wind_profile_cube: cli.inputcube,
    high_res_orog_cube: cli.inputcube,
    model_orog_cube: cli.inputcube,
    model_orog_stddev_cube: cli.inputcube,
    model_silhouette_roughness_cube: cli.inputcube,
    landmask_cube: cli.inputcube,
    *,
    target_height_levels: cli.comma_separated_list = None,
):
    """Apply unresolved-orography wind-speed corrections.

    Args:
        target_wind_speed_cube:
            Wind-speed cube to be corrected.

        wind_profile_cube:
            Wind-speed profile cube used to fit roughness and reference-wind
            parameters. Must contain wind speeds on heights between ground and
            300 m above ground level. Can be the same cube as ``target_wind_speed_cube``.

        high_res_orog_cube:
            High-resolution orography cube.

        model_orog_cube:
            Model-resolution orography cube.

        model_orog_stddev_cube:
            Sub-grid orography standard deviation cube.

        model_silhouette_roughness_cube:
            Sub-grid silhouette roughness cube.

        landmask_cube:
            Land-sea mask cube.

        target_height_levels:
            Comma-separated list of target heights above ground level, in
            metres. Use ``None`` to apply corrections at the heights already
            present on ``target_wind_speed_cube``.

    Returns:
        iris.cube.Cube:
            Wind-speed cube with unresolved-orography correction applied.

    Raises:
        ValueError:
            If the number of realizations on ``wind_profile_cube`` and
            ``target_wind_speed_cube`` are incompatible and cannot be
            broadcast.
    """
    import iris
    from iris.exceptions import CoordinateNotFoundError

    from improver.wind_calculations.wind_downscaling import WindDownscaling

    if target_height_levels is None:
        parsed_target_height_levels = None
    elif isinstance(target_height_levels, list):
        if (
            len(target_height_levels) == 1
            and str(target_height_levels[0]).lower() == "none"
        ):
            parsed_target_height_levels = None
        else:
            parsed_target_height_levels = [
                float(value) for value in target_height_levels
            ]
    elif isinstance(target_height_levels, str):
        if target_height_levels.lower() == "none":
            parsed_target_height_levels = None
        else:
            parsed_target_height_levels = [
                float(value) for value in target_height_levels.split(",")
            ]
    else:
        parsed_target_height_levels = target_height_levels

    plugin = WindDownscaling(
        high_res_orog_cube,
        model_orog_cube,
        model_orog_stddev_cube,
        model_silhouette_roughness_cube,
        landmask_cube,
    )

    try:
        wind_profile_slices = list(wind_profile_cube.slices_over("realization"))
    except CoordinateNotFoundError:
        wind_profile_slices = [wind_profile_cube]

    try:
        target_wind_speed_slices = list(
            target_wind_speed_cube.slices_over("realization")
        )
    except CoordinateNotFoundError:
        target_wind_speed_slices = [target_wind_speed_cube]

    n_profile = len(wind_profile_slices)
    n_target = len(target_wind_speed_slices)

    if n_profile != n_target:
        if n_profile == 1:
            wind_profile_slices = wind_profile_slices * n_target
        elif n_target == 1:
            target_wind_speed_slices = target_wind_speed_slices * n_profile
        else:
            try:
                profile_by_realization = {
                    float(cube.coord("realization").points[0]): cube
                    for cube in wind_profile_slices
                }
                target_by_realization = {
                    float(cube.coord("realization").points[0]): cube
                    for cube in target_wind_speed_slices
                }
            except CoordinateNotFoundError:
                raise ValueError(
                    "Mismatch in realization count between wind_profile_cube "
                    "and target_wind_speed_cube."
                )

            common_realizations = sorted(
                set(profile_by_realization).intersection(target_by_realization)
            )
            if not common_realizations:
                raise ValueError(
                    "Mismatch in realization count between wind_profile_cube "
                    "and target_wind_speed_cube."
                )

            wind_profile_slices = [
                profile_by_realization[realization]
                for realization in common_realizations
            ]
            target_wind_speed_slices = [
                target_by_realization[realization]
                for realization in common_realizations
            ]

    wind_speed_list = iris.cube.CubeList()
    for wind_profile_slice, target_wind_speed_slice in zip(
        wind_profile_slices,
        target_wind_speed_slices,
    ):
        result = plugin(
            wind_profile_slice,
            target_wind_speed_slice,
            target_height_levels=parsed_target_height_levels,
        )
        wind_speed_list.append(result)

    wind_speed = wind_speed_list.merge_cube()

    non_dim_coords = [x.name() for x in wind_speed.coords(dim_coords=False)]
    if "realization" in non_dim_coords:
        wind_speed = iris.util.new_axis(wind_speed, "realization")

    return wind_speed
