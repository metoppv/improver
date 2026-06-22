from typing import Literal


class RainForestsModelConfig(
    dict[
        # lead time in hours, (integer converted to str)
        str,
        dict[
            # threshold in metres (formatted to "08.6f")
            str,
            dict[
                # literal "lightgbm_model" or "treelite_model"
                Literal["lightgbm_model", "treelite_model"],
                # path to output file
                str,
            ],
        ],
    ]
):
    """Define the config structure used to describe the high-level RainForests model structure.

    RainForestsModelConfig is a nested dictionar of format:

    .. code-block:: json

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

    - top level key describes the lead-hour,
    - next level key describes the threshold,
    - corresponding values locate the associated model file.
    """

    def __init__(self, *arg, **kw):
        super(RainForestsModelConfig, self).__init__(*arg, **kw)
