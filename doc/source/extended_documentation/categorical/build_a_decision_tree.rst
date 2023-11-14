**Decision trees**

Decision trees use diagnostic fields to diagnose a suitable category to represent
the weather conditions, such as a weather symbol. The tree is comprised
of a series of interconnected decision nodes, leaf nodes and a stand-alone meta node.
At each decision node one or multiple forecast diagnostics are compared to
predefined threshold values. The decision node has an if_true and if_false path on
to the next node. By traversing the nodes it should be possible, given the right
weather conditions, to arrive at any of the leaf nodes, which describe the leaf
name, code, and optional information for night-time and modal grouping.

The first few nodes of a decision tree are represented in the schematic below.

.. figure:: extended_documentation/categorical/thunder_nodes.png
     :align: center
     :scale: 80 %
     :alt: Schematic of thundery nodes in a decision tree

There are two thresholds being used in these nodes. The first is the diagnostic
threshold which identifies the critical value for a given diagnostic. In the
first node of the schematic shown this threshold is the count of lightning
flashes in an hour exceeding 0.0. The second threshold is the probability of
exceeding (in this case) this diagnostic threshold. In this first node it's a
probability of 0.3 (30%). So the node overall states that if there is an equal
or greater than 30% probability of any lightning flashes in the hour being
forecast, proceed to the if_true node, else move to the if_false node.

**Encoding a decision tree**

The meta node provides the name to use for the metadata of the resulting cube and
can be anywhere in the decision tree, but must have "meta" as its key.
This becomes the cube name and is also used for two attributes that describe the
categorical data: **<name>** and **<name>_meaning**::

  {
    "meta": {
        "name": "weather_code",
    },
  }


The first decision node in the thundery nodes shown above is encoded as follows::

  {
    "lightning": {
        "if_true": "lightning_cloud",
        "if_false": "heavy_precipitation",
        "if_diagnostic_missing": "if_false",
        "probability_thresholds": [0.3],
        "threshold_condition": ">=",
        "condition_combination": "",
        "diagnostic_fields": [
            "probability_of_number_of_lightning_flashes_per_unit_area_in_vicinity_above_threshold"
        ],
        "diagnostic_thresholds": [[0.0, "m-2"]],
        "diagnostic_conditions": ["above"]
    },
  }

The key at the first level, "lightning" in this case, names the node so that it
can be targeted as an if_true or if_false destination from other nodes. The dictionary
accessed with this key contains the essentials that make the node function.

  - **if_true** (str): The next node if the condition in this
    node is true.
  - **if_false** (str): The next node if the condition in this node
    is false.
  - **if_diagnostic_missing** (str, optional): If the expected
    diagnostic is not provided, should the tree proceed to the if_true or if_false
    node. This can be useful if the tree is to be applied to output from
    different models, some of which do not provide all the diagnostics that might
    be desirable.
  - **probability_thresholds** (list(float)): The probability threshold(s) that
    must be exceeded or not exceeded (see threshold_condition) for the node to
    progress to the succeed target. Two values required if condition_combination
    is being used.
  - **threshold_condition** (str): Defines the inequality test to be applied to
    the probability threshold(s). Inequalities that can be used are "<=", "<",
    ">", ">=".
  - **condition_combination** (str): If multiple tests are being applied in a
    single node, this value determines the logic with which they are combined.
    The values can be "AND", "OR".
  - **diagnostic_fields** (List(str or List(str)): The name(s) of the
    diagnostic(s) that will form the test condition in this node. There may be
    multiple diagnostics if they are being combined in the test using a
    condition_combination. Alternatively, if they are being manipulated within
    the node (e.g. added together), they must be separated by the desired
    operators (e.g. 'diagnostic1', '+', 'diagnostic2').
  - **diagnostic_thresholds** (List(List(float, str, Optional(int))): The
    diagnostic threshold value and units being used in the test. An optional
    third value provides a period in seconds that is associated with the
    threshold value. For example, a precipitation accumulation threshold might
    be given for a 1-hour period (3600 seconds). If instead the decision tree
    generates data representing a 3-hour period
    using 3-hour precipitation accumulations then the threshold
    value will be scaled up by a factor of 3. Only thresholds with an
    associated period will be scaled in this way. A threshold [value, units] pair
    must be provided for each diagnostic field with the same nested list structure;
    as the basic unit is a list of value and unit, the overall nested structure is
    one list deeper.
  - **diagnostic_conditions** (as diagnostic_fields): The expected inequality
    that has been used to construct the input probability field. This is checked
    against the spp__relative_to_threshold attribute of the threshold coordinate
    in the provided diagnostic.

The first leaf node above is encoded as follows::

  {
    "Thunder_Shower_Day": {
        "leaf": 29,
        "if_night": "Thunder_Shower_Night",
        "group": "convection",
        "is_unreachable": True,
    },
  }

The key at the first level, "Thunder_Shower_Day" in this case, names the node so that it
can be targeted as an if_true or if_false destination from decision nodes. The key
also forms part of the metadata attribute defining the category meanings. The dictionary
accessed with this key contains the following.

  - **leaf** (int): The category code associated with this leaf
  - **if_night** (str, optional): The alternate leaf node to be used when a night
    time symbol is required.
  - **group** (str, optional): Indicates which group this leaf belongs to when
    determining the modal category.
  - **is_unreachable** (bool): True for a leaf which needs including in the meta data but
    cannot be reached.

The modal category also relies on the severity of symbols generally increasing with
the category value, so that in the case of ties, the more severe category is selected.

Every decision tree must have a starting node, and this is taken as the first
node defined in the dictionary, or second if the first node is the meta node.

Manipulation of the diagnostics is possible using the decision tree configuration
to enable more complex comparisons. For example::

  "heavy_rain_or_sleet_shower": {
      "if_true": 14,
      "if_false": 17,
      "probability_thresholds": [0.0],
      "threshold_condition": "<",
      "condition_combination": "",
      "diagnostic_fields": [
          [
              "probability_of_lwe_sleetfall_rate_above_threshold",
              "+",
              "probability_of_lwe_snowfall_rate_above_threshold",
              "-",
              "probability_of_rainfall_rate_above_threshold"
          ]
      ],
      "diagnostic_thresholds": [[[1.0, "mm hr-1"], [1.0, "mm hr-1"], [1.0, "mm hr-1"]]],
      "diagnostic_conditions": [["above", "above", "above"]]
  },

This node uses three diagnostics. It combines them according to the mathematical
operators that separate the names in the `diagnostic_fields` list. The resulting
value is compared to the probability threshold value using the threshold condition.
In this example the purpose is to check whether the probability of the rain rate
exceeding 1.0 mm/hr is greater than the combined probability of the same rate
being exceeded by sleet and snow.
