# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.

"""RainForests model compilation plugin."""

from improver import BasePlugin
from improver.calibration import (
    treelite_packages_available,
)

LIGHTGBM_EXTENSION = ".txt"
TREELITE_EXTENSION = ".so"


class CompileRainForestsModel(BasePlugin):
    """Class to compile RainForests tree models"""

    def __init__(self, toolchain="gcc", verbose=False, parallel_comp=0):
        """Initialise the options used when compiling models.

        Args:
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
        """

        self.treelite_available = treelite_packages_available()
        if not self.treelite_available:
            raise ModuleNotFoundError("Could not find TreeLite module")

        self.toolchain = toolchain
        self.verbose = verbose
        self.treelight_params = {"parallel_comp": parallel_comp, "quantize": 1}

    def process(self, model_file, output_dir):
        """Compile a lightgbm model with Treelite.

        Args:
            model_file (pathlib.Path):
                Path to LightGBM Booster file.
            output_dir (pathlib.Path):
                Directory where the compiled Treelite predictor file will be created.
        """

        import tl2cgen
        import treelite

        # Input validation
        if model_file.suffix.lower() != LIGHTGBM_EXTENSION:
            raise ValueError(f"Input path must have the extension {LIGHTGBM_EXTENSION}")
        if not output_dir.is_dir():
            raise ValueError("Output path must be a directory")

        output_filepath = output_dir / f"{model_file.stem}{TREELITE_EXTENSION}"

        model = treelite.frontend.load_lightgbm_model(model_file)

        tl2cgen.export_lib(
            model,
            libpath=output_filepath,
            toolchain=self.toolchain,
            verbose=self.verbose,
            params=self.treelight_params,
        )
