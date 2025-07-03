# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Utility classes used in unit tests for rainforests calibration."""


class MockBooster:
    def __init__(self, model_file, **kwargs):
        self.model_class = "lightgbm-Booster"
        self.model_file = model_file

    def reset_parameter(self, params):
        self.threads = params.get("num_threads")
        return self


class MockPredictor:
    def __init__(self, libpath, nthread, **kwargs):
        self.model_class = "treelite-Predictor"
        self.threads = nthread
        self.model_file = libpath
