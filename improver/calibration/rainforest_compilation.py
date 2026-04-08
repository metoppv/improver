# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

"""RainForests model compilation plugin."""

from pathlib import Path

from improver import BasePlugin
from improver.calibration import (
    treelite_packages_available,
)

LIGHTGBM_EXTENSION = ".txt"
TREELITE_EXTENSION = ".so"


class CompileRainForestsModel(BasePlugin):
    """Class to compile RainForests tree models"""

    def __init__(
        self,
        model_config_dict: dict[int, dict[str, dict[str, str]]],
        toolchain="gcc",
        verbose=False,
        parallel_comp=0,
    ):
        """Initialise the options used when compiling models.

        Args:
            model_config_dict:
                Dictionary containing Rainforests model configuration variables.
            toolchain (str):
                Toolchain to use for Treelite model compilation.
                'gcc' (default), 'msvc', 'clang' or a specific variation of clang or gcc
                (e.g. 'gcc-7').
            verbose (bool):
                Print verbose output during compilation
            parallel_comp (int):
                Enables parallel compilation to reduce time and memory consumption.
                Value is the number of processes to use.
                Defaults to 0 (no parallel compilation)

        Dictionary is of format::

        {
        "24": {
            "0.000010": {
                "lightgbm_model": "<path_to_lightgbm_model_object>",
                "treelite_model": "<path_to_treelite_model_object>"
            },
            "0.000050": {
                "lightgbm_model": "<path_to_lightgbm_model_object>",
                "treelite_model": "<path_to_treelite_model_object>"
            },
            "0.000100": {
                "lightgbm_model": "<path_to_lightgbm_model_object>",
                "treelite_model": "<path_to_treelite_model_object>"
            },
        }

        The keys specify the lead times and model threshold values, while the
        associated values are the path to the corresponding tree-model objects
        for that lead time and threshold.
        """

        treelite_available = treelite_packages_available()
        if not treelite_available:
            raise ModuleNotFoundError("Could not find TreeLite module")

        self.config = model_config_dict

        self.toolchain = toolchain
        self.verbose = verbose
        self.treelight_params = {"parallel_comp": parallel_comp, "quantize": 1}

    def process(self, allow_missing=False):
        """Compile all configured LightGBM models with Treelite.

            Args:
                allow_missing (bool):
                    If False (default), throws if any LightGBM models are missing.
                    If True, any missing LightGBM files will be ignored.

        Iterates through all lead times and thresholds in the model config dictionary
        and compiles the corresponding LightGBM models to Treelite predictors.
        """

        for lead_time_config in self.config.values():
            for threshold_data in lead_time_config.values():
                self._compile_model(
                    Path(threshold_data["lightgbm_model"]),
                    Path(threshold_data["treelite_model"]),
                    allow_missing,
                )

    def _compile_model(self, lightgbm_path, output_path, allow_missing=False):
        """Compile a lightgbm model with Treelite.

        Args:
            lightgbm_path (pathlib.Path):
                Path to LightGBM Booster file.
            output_path (pathlib.Path):
                Path where the compiled Treelite predictor file will be created.
        """

        import tl2cgen
        import treelite

        # Validate both paths
        if not lightgbm_path.is_file():
            if allow_missing:
                return
            else:
                raise ValueError(f"Model file not found: {lightgbm_path}")

        if lightgbm_path.suffix.lower() != LIGHTGBM_EXTENSION:
            raise ValueError(f"Input path must have extension {LIGHTGBM_EXTENSION}")
        if output_path.suffix.lower() != TREELITE_EXTENSION:
            raise ValueError(f"Output path must have extension {TREELITE_EXTENSION}")

        Path.mkdir(output_path.parent, parents=True, exist_ok=True)

        model = treelite.frontend.load_lightgbm_model(lightgbm_path)

        tl2cgen.export_lib(
            model,
            libpath=output_path,
            toolchain=self.toolchain,
            verbose=self.verbose,
            params=self.treelight_params,
        )
