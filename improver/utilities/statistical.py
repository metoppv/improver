# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Module to contain statistical methods."""
import numpy as np

from improver import BasePlugin
from copy import deepcopy
from pygam import GAM, l, s, te, f
from typing import Dict


class GAMFit(BasePlugin):
    """
    Class for fitting Generalized Additive Models (GAMs) which predict the mean and standard deviation of input
    forecasts or observations.

    This class uses functionality from pyGAM (https://pygam.readthedocs.io/en/latest/index.html), which is used for
    fitting the model.
    """

    def __init__(
        self,
        model_specification: Dict,
        max_iter: int = 100,
        tol: float = 0.0001,
        distribution: str = 'normal',
        link: str = 'identity',
        fit_intercept: bool = True,
    ):
        """
        Initialize class for fitting GAMs using pyGAM.

        Args:
            model_specification:
                a dictionary with arbitrary keys and values a list of three items (in order):
                    1. a string containing a single pyGAM term; one of 'l' (linear), 's' (spline), 'te' (tensor), or
                    'f' (factor)
                    2. a list of integers which correspond to the features to be included in that term
                    3. a dictionary of kwargs to be included when defining the term
            max_iter:
                a pyGAM argument which determines the maximum iterations allowed when fitting the GAM
            tol:
                a pyGAM argument determining the tolerance used to define the stopping criteria
            distribution:
                a pyGAM argument determining the distribution to be used in the model
            link:
                a pyGAM argument determining the link function to be used in the model
            fit_intercept:
                a pyGAM argument determining whether to include an intercept term in the model
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
        for index, (key, values) in enumerate(self.model_specification.items()):
            # For each key in the dictionary, parse the value to create a pyGAM term from that value.
            # The first term in the dictionary value defines the type of term, the second defines which variables are
            # included in that term, and the third contains a dictionary of kwargs.
            if values[0] == "l":  # linear term
                new_term = l(*values[1], **values[2])
            elif values[0] == "s":  # spline term
                new_term = s(*values[1], **values[2])
            elif values[0] == "te":  # tensor term
                new_term = te(*values[1], **values[2])
            elif values[0] == "f":  # factor term
                new_term = f(*values[1], **values[2])
            else:
                msg = (
                    f"An unrecognised term has been included in the GAM model specification. The term was {values[0]},"
                    f" the accepted terms are l, s, te, f.")
                raise ValueError(msg)

            if index == 0:
                # initialise the equation variable
                eqn = deepcopy(new_term)
            else:
                # add new term to the existing equation
                eqn += new_term

        return eqn

    def process(self, X: np.ndarray, y: np.ndarray):
        """
        Fit a GAM model using pyGAM.

        Args:
            X: An array of predictors
            y: An array of target values associated with the predictors in X

        Returns:
            A fitted pyGAM GAM model.
        """
        eqn = self.create_pygam_model()
        gam = GAM(
            eqn,
            max_iter=self.max_iter,
            tol=self.tol,
            distribution=self.distribution,
            link=self.link,
            fit_intercept=self.fit_intercept
        ).fit(X, y)

        return gam


class GAMPredict(BasePlugin):
    """Class for predicting new outputs from a fitted GAM given new input variables."""
    def __init__(self):
        """Initialize class"""
    def process(self, gam, X: np.ndarray) -> np.ndarray:
        """
        Use pyGAM functionality to predict values from a fitted GAM.

        Args:
            gam: Fitted pyGAM GAM model
            X: An array of inputs to use to predict new values. Must have the same number of columns as used for
            training.

        Returns:
            A 1-D array of values predicted by the GAM.
        """
        return gam.predict(X)
