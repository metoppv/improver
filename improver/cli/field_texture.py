# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of improver and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Script to calculate whether or not the input field texture exceeds a given threshold."""

from improver import cli


@cli.clizefy
@cli.with_output
def process(
    cube: cli.inputcube,
    *,
    nbhood_radius: float = 20000.0,
    textural_threshold: float = 0.05,
    diagnostic_threshold: float = 0.8125,
    model_id_attr: str = None,
):

    """Calculates field texture for a given neighbourhood radius.

       The field texture is an assessment of the transitions/edges within a
       neighbourhood of a grid point to indicate whether the field is rough
       or smooth.

    Args:
        cube (iris.cube.Cube):
            The diagnostic for which texture is to be assessed. For example cloud
            area fraction where transitions between cloudy regions and cloudless
            regions will be diagnosed. Defaults set assuming cloud area fraction
            cube.

        nbhood_radius (float):
            The neighbourhood radius in metres within which the number of potential
            transitions should be calculated. This forms the denominator in the
            calculation of the ratio of actual to potential transitions that indicates a
            field's texture. A larger radius should be used for diagnosing larger-scale
            textural features. Default value set to 10000.0 assuming cloud area fraction
            cube.

        textural_threshold (float):
            A unit-less threshold value that defines the ratio value above which
            the field is considered rough and below which the field is considered
            smoother. Default value set to 0.05 assuming cloud area fraction cube.

        diagnostic_threshold (float):
            The diagnostic threshold for which field texture will be calculated.
            A ValueError is raised if this threshold is not present on the input
            cube. Default value set to 0.8125 corresponding to 6 oktas, assuming
            cloud area fraction cube.

        model_id_attr (str):
            Name of the attribute used to identify the source model for
            blending.

    Returns:
        iris.cube.Cube:
            A field texture cube containing values between 0 and 1, where 0
            indicates no transitions and 1 indicates the maximum number of
            possible transitions has been achieved, within the neighbourhood of
            the grid point.
    """

    from improver.utilities.textural import FieldTexture

    field_texture = FieldTexture(
        nbhood_radius,
        textural_threshold,
        diagnostic_threshold,
        model_id_attr=model_id_attr,
    )(cube)
    return field_texture
