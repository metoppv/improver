from typing import Literal, TypeAlias

"""Define the config structure used to describe the high-level RainForests model structure;

RainForestsModelConfig dictionary is of format:

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

# Definition for the model config object, and json file
RainForestsModelConfig: TypeAlias = dict[
    str,  # lead time in hours, (integer converted to str)
    dict[
        str,  # threshold in metres (formatted to "08.6f")
        dict[
            Literal[
                "lightgbm_model", "treelite_model"
            ],  # literal "lightgbm_model" or "treelite_model"
            str,  # path to output file
        ],
    ],
]
