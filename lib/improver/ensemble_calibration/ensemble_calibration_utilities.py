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
This module defines all the utilities used by the "plugins"
specific for ensemble calibration.

"""
import iris
import numpy as np

from improver.utilities.cube_manipulation import merge_cubes


def convert_cube_data_to_2d(
        forecast, coord="realization", transpose=True):
    """
    Function to convert data from a N-dimensional cube into a 2d
    numpy array. The result can be transposed, if required.

    Args:
        forecast (iris.cube.Cube):
            N-dimensional cube to be reshaped.
        coord (str):
            The data will be flattened along this coordinate.
        transpose (bool):
            If True, the resulting flattened data is transposed.
            This will transpose a 2d array of the format [:, coord]
            to [coord, :].
            If False, the resulting flattened data is not transposed.
            This will result in a 2d array of format [:, coord].

    Returns:
        numpy.ndarray:
            Reshaped 2d array.

    """
    forecast_data = []
    if np.ma.is_masked(forecast.data):
        forecast.data = np.ma.filled(forecast.data, np.nan)

    for coord_slice in forecast.slices_over(coord):
        forecast_data.append(coord_slice.data.flatten())
    if transpose:
        forecast_data = np.asarray(forecast_data).T
    return np.array(forecast_data)


def flatten_ignoring_masked_data(data_array, preserve_leading_dimension=False):
    """
    Flatten an array, selecting only valid data if the array is masked. There
    is also the option to reshape the resulting array so it has the same
    leading dimension as the input array, but the other dimensions of the
    array are flattened. It is assumed that each of the slices
    along the leading dimension are masked in the same way. This functionality
    is used in EstimateCoefficientsForEnsembleCalibration when realizations
    are used as predictors.

    Args:
        data_array (numpy.ndarray or numpy.ma.MaskedArray):
            An array or masked array to be flattened. If it is masked and the
            leading dimension is preserved the mask must be the same for every
            slice along the leading dimension.
        preserve_leading_dimension (bool):
            Default False.
            If True the flattened array is reshaped so it has the same leading
            dimension as the input array. If False the returned array is 1D.
    Returns:
        numpy.ndarray:
            A flattened array containing only valid data. Either 1D or, if
            preserving the leading dimension 2D. In the latter case the
            leading dimension is the same as the input data_array.
    Raises:
        ValueError: If preserving the leading dimension and the mask on the
                    input array is not the same for every slice along the
                    leading dimension.
    """
    if np.ma.is_masked(data_array):
        # If we have multiple 2D x-y slices check that the mask is the same for
        # each slice along the leading dimension.
        if data_array.ndim > 2:
            first_slice_mask = data_array[0].mask

            for i in range(1, data_array.shape[0]):
                if not np.all(first_slice_mask == data_array[i].mask):
                    msg = ("The mask on the input array is not the same for "
                           "every slice along the leading dimension.")
                    raise ValueError(msg)
        # If the mask is ok, select the unmasked data, flattening it at
        # the same time.
        result = data_array[~data_array.mask]
    else:
        result = data_array.flatten()
    if preserve_leading_dimension:
        # Reshape back to give the same leading dimension in the array. The 2nd
        # dimension is inferred through the use of -1.
        final_shape = (data_array.shape[0], -1)
        result = result.reshape(final_shape)
    return result


def check_predictor_of_mean_flag(predictor_of_mean_flag):
    """
    Check the predictor_of_mean_flag at the start of the
    process methods in relevant ensemble calibration plugins,
    to avoid having to check and raise an error later.

    Args:
        predictor_of_mean_flag (str):
            String to specify the input to calculate the calibrated mean.
            Currently the ensemble mean ("mean") and the ensemble realizations
            ("realizations") are supported as the predictors.

    Raises:
        ValueError: If the predictor_of_mean_flag is not valid.
    """
    if predictor_of_mean_flag.lower() not in ["mean", "realizations"]:
        msg = ("The requested value for the predictor_of_mean_flag {} "
               "is not an accepted value."
               "Accepted values are 'mean' or 'realizations'").format(
                   predictor_of_mean_flag.lower())
        raise ValueError(msg)


class SplitHistoricForecastAndTruth():

    """Split the historic forecasts and truth datasets based on the
    metadata identifiers provided."""

    def __init__(self, historic_forecast_dict, truth_dict):
        """Initialise the plugin.

        Args:
            historic_forecast_dict (dict):
                Dictionary specifying the metadata that defines the historic
                forecast. For example:
                ::

                    {
                        "attributes": {
                            "mosg__model_configuration": "uk_ens"
                        }
                    }
            truth_dict (dict):
                Dictionary specifying the metadata that defines the truth.
                For example:
                ::

                    {
                        "attributes": {
                            "mosg__model_configuration": "uk_det"
                        }
                    }

        Raises:
            NotImplementedError:
                'attributes' is the only supported key for the input
                dictionaries.
        """
        self.historic_forecast_dict = historic_forecast_dict
        self.truth_dict = truth_dict

        for input_dict in [self.historic_forecast_dict, self.truth_dict]:
            for key in input_dict.keys():
                if key != "attributes":
                    msg = ("At present, 'attributes' is the only supported "
                           "key for the input. Support for specifying other "
                           "metadata can be added if required.")
                    raise NotImplementedError(msg)

    def __repr__(self):
        """Represent the plugin instance as a string."""
        result = ('<SplitHistoricForecastsAndTruth: '
                  'historic_forecast_dict={}, truth_dict={}>')
        return result.format(self.historic_forecast_dict, self.truth_dict)

    @staticmethod
    def _find_required_cubes_using_metadata(cubes, input_dict):
        """
        Extract the cube that matches the information within the input
        dictionary.

        Args:
            cubes (iris.cube.CubeList):
                The cubes that will be checked for matches against the
                metadata specified in the input_dict.
            input_dict (dict):
                A dictionary containing the metadata that will be used to
                identify the desired cubes.

        Returns:
            iris.cube.CubeList:
                CubeList containing cubes that match the metadata supplied
                within the input dictionary.

        Raises:
            ValueError:
                The metadata supplied resulted in no matching cubes.

        """
        constr = iris.AttributeConstraint(**input_dict["attributes"])
        cubelist = cubes.extract(constr)
        if not cubelist:
            msg = ("The metadata to identify the desired historic forecast or "
                   "truth has found nothing matching the metadata information "
                   "supplied: {}".format(input_dict))
            raise ValueError(msg)
        return cubelist

    def process(self, cubes):
        """
        Separate the input cubes into the historic_forecasts and truth based
        on the metadata information supplied within the input dictionaries.

        Args:
             cubes (iris.cube.CubeList):
                CubeList of input cubes that are expected to contain a mixture
                of historic forecasts and truth.

        Returns:
            (tuple): tuple containing:
                iris.cube.Cube:
                    A cube containing the historic forecasts.
                iris.cube.Cube:
                    A cube containing the truth datasets.
        """
        historic_forecasts = self._find_required_cubes_using_metadata(
            cubes, self.historic_forecast_dict)
        truths = self._find_required_cubes_using_metadata(
            cubes, self.truth_dict)
        # Use improver merge_cubes to equalise attributes
        return merge_cubes(historic_forecasts), merge_cubes(truths)
