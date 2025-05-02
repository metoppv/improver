# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the GAMFit class within statistical.py"""
import numpy as np
import pytest

from improver.utilities.statistical import GAMFit
from pygam import GAM, l, s, te, f
from pygam.datasets import wage


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "model_specification": {"term": ["l", [0], {}]}
        },  # define a model specification but leave all other inputs as default
        {
            "model_specification": {"term": ["l", [0], {}]},
            "max_iter": 200,
            "tol": 0.1,
        },  # check that inputs related to model fitting are initialised correctly
        {
            "model_specification": {"term": ["l", [0], {}]},
            "distribution": "gamma",
            "link": "inverse",
            "fit_intercept": False,
        },  # check that inputs related to the GAM model design are initialised correctly

    ]
)
def test__init__(kwargs):
    """Test that the class initialises variables correctly."""
    # Define the default, then update with any differently specified inputs
    expected = {
        "model_specification": None,
        "max_iter": 100,
        "tol": 0.0001,
        "distribution": "normal",
        "link": "identity",
        "fit_intercept": True
    }
    expected.update(kwargs)
    result = GAMFit(**kwargs)

    for key in [key for key in kwargs.keys()]:
        assert getattr(result, key) == kwargs[key]


@pytest.mark.parametrize(
    "test,model_specification",
    [
        (
            "basic",
            {
                 "term_1": ["l", [0], {}],
                 "term_2": ["s", [1], {}],
                 "term_3": ["te", [2, 3], {}],
                 "term_4": ["f", [4], {}]
            },
        ),  # test that each type of GAM term can be created correctly
        (
            "with_kwargs",
            {
                "term_1": ["l", [0], {"lam": 0.8}],
                "term_2": ["s", [1], {"n_splines": 10, "basis": "cp"}],
            },
        ),  # test that kwargs are passed to the pyGAM terms correctly
        (
            "exception",
            {
                "term_1": ["l", [0], {}],
                "term_2": ["kittens", [1], {}],
            },
        )  # test that an exception is raised when an unknown term is provided
    ]
)
def test_create_pygam_model(test, model_specification):
    """Test that this method correctly creates a pyGAM equation and raises an exception when
    provided with a bad input."""
    if test in ["basic", "with_kwargs"]:
        if test is "basic":
            expected = l(0) + s(1) + te(2, 3) + f(4)
        elif test is "with_kwargs":
            expected = l(0, lam=0.8) + s(1, n_splines=10, basis="cp")
        result = GAMFit(model_specification).create_pygam_model()
        assert result == expected
    elif test is "exception":
        expected_msg = "An unrecognised term has been included in the GAM model specification."
        with pytest.raises(ValueError, match=expected_msg):
            GAMFit(model_specification).create_pygam_model()


def test_process():
    """Test that the process method returns the expected results. Uses an example from the pyGAM quick start
    documentation: https://pygam.readthedocs.io/en/latest/notebooks/quick_start.html#Fit-a-Model."""
    X, y = wage()
    model_specification = {"term_1": ["s", [0], {}], "term_2": ["s", [1], {}], "term_3": ["f", [2], {}]}

    expected = GAM(s(0) + s(1) + f(2)).fit(X, y)
    result = GAMFit(model_specification).process(X, y)

    for i, term in enumerate(result.terms):
        # for each non-intercept term in each fitted GAM, compare their partial dependence values and confidence
        # intervals, which should be equal if the fitted models are the same.
        if term.isintercept:
            continue

        XX = result.generate_X_grid(term=i)
        result_pdep, result_confi = result.partial_dependence(term=i, X=XX, width=0.95)
        expected_pdep, expected_confi = expected.partial_dependence(term=i, X=XX, width=0.95)

        assert np.array_equal(result_pdep, expected_pdep)
        assert np.array_equal(result_confi, expected_confi)

    for key in list(result.statistics_.keys()):
        # check that other features of the fitted GAM models are equal.
        if key in [
            "n_samples", "m_features", "edof", "scale", "AIC", "AICc", "GCV", "UBRE", "loglikelihood", "deviance"
        ]:
            assert result.statistics_[key] == expected.statistics_[key]
        else:
            assert np.array_equal(result.statistics_[key], expected.statistics_[key])
