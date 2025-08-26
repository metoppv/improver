# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module to contain methods for fitting and predicting using generalized additive
models."""

from copy import deepcopy
from typing import Any, List

import numpy as np

from improver import BasePlugin


class GAMFit(BasePlugin):
    """
    Class for fitting Generalized Additive Models (GAMs) which predict the mean or
    standard deviation of input forecasts or observations.

    This class uses functionality from pyGAM
    (https://pygam.readthedocs.io/en/latest/index.html) to fit the model.
    """

    def __init__(
        self,
        model_specification: List[List[Any]],
        max_iter: int = 100,
        tol: float = 0.0001,
        distribution: str = "normal",
        link: str = "identity",
        fit_intercept: bool = True,
    ):
        """
        Initialize class for fitting GAMs using pyGAM.

        Args:
            model_specification:
                A list containing lists of three items (in order):
                    1. a string containing a single pyGAM term; one of 'linear',
                    'spline', 'tensor', or 'factor'
                    2. a list of indices of the features to be included in that term,
                    corresponding to the index of those features in the predictor array
                    3. a dictionary of kwargs to be included when defining the term
            max_iter:
                A pyGAM argument which determines the maximum iterations allowed when
                fitting the GAM. Defaults to 100.
            tol:
                A pyGAM argument determining the tolerance used to define the stopping
                criteria. Defaults to 0.0001.
            distribution:
                A pyGAM argument determining the distribution to be used in the model.
                The default is a normal distribution.
            link:
                A pyGAM argument determining the link function to be used in the model.
                Defaults to the identity link function, which implies a direct
                relationship between predictors and target.
            fit_intercept:
                A pyGAM argument determining whether to include an intercept term in
                the model. Default is True.
        """
        self.model_specification = model_specification
        self.max_iter = max_iter
        self.tol = tol
        self.distribution = distribution
        self.link = link
        self.fit_intercept = fit_intercept

    def create_pygam_model(self):
        """
        Create a GAM model using pyGAM from the model_specification dictionary.

        Returns:
            GAM model equation constructed using pyGAM model terms.
        """
        # Import from pygam here to minimize dependencies
        from pygam import f, l, s, te

        term = {
            "factor": f,
            "linear": l,
            "spline": s,
            "tensor": te,
        }  # create dictionary of permissible pyGAM model terms

        for index, config in enumerate(self.model_specification):
            # For each config in the list, parse the config to create a pyGAM term
            # from that config. The first term in the config defines the type of term,
            # the second defines which variables are included in that term, and the
            # third contains a dictionary of kwargs.
            if config[0] in term.keys():
                new_term = term[config[0]](*config[1], **config[2])
            else:
                msg = (
                    f"An unrecognised term has been included in the GAM model "
                    f"specification. The term was {config[0]}, the accepted terms are "
                    f"linear, spline, tensor, factor."
                )
                raise ValueError(msg)

            if index == 0:
                # Initialize the equation variable
                eqn = deepcopy(new_term)
            else:
                # Add new term to the existing equation
                eqn += new_term

        return eqn

    def process(self, predictors: np.ndarray, targets: np.ndarray):
        """
        Fit a GAM model using pyGAM.

        Args:
            predictors: A 2-D array of predictors. The index of each column (feature) is
             used in model_specification to determine which feature is included in each
             model term.
            targets: A 1-D array of target values associated with the predictors.

        Returns:
            A fitted pyGAM GAM model.
        """
        # Monkey patch for pyGAM due to handling of sparse arrays in some versions of
        # scipy.
        import scipy.sparse

        def to_array(self):
            return self.toarray()

        scipy.sparse.spmatrix.A = property(to_array)
        # Import from pygam here to minimize dependencies
        from pygam import GAM

        eqn = self.create_pygam_model()
        gam = GAM(
            eqn,
            max_iter=self.max_iter,
            tol=self.tol,
            distribution=self.distribution,
            link=self.link,
            fit_intercept=self.fit_intercept,
        ).fit(predictors, targets)

        return gam


class GAMPredict(BasePlugin):
    """Class for predicting new outputs from a fitted GAM given new input predictors."""

    def __init__(self):
        """Initialize class"""

    def process(self, gam, predictors: np.ndarray) -> np.ndarray:
        """
        Use pyGAM functionality to predict values from a fitted GAM.

        Args:
            gam: A fitted pyGAM GAM model.
            predictors: A 2-D array of inputs to use to predict new values. Each
                feature (column) should have the same index as in the training dataset.

        Returns:
            A 1-D array of values predicted by the GAM with each value in the array
            corresponding to one row in the input predictors.
        """

        return gam.predict(predictors)
