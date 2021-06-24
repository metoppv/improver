**Weather symbol decision trees**

Weather symbol decision trees use diagnostic fields to diagnose a suitable
symbol to represent the weather conditions. The tree is comprised of a series
of interconnected decision nodes. At each node one or multiple forecast
diagnostics are compared to predefined threshold values. The node has a success
and failure path on to the next node, or on to a resulting weather symbol. By
traversing the nodes it should be possible, given the right weather conditions,
to arrive at any of the weather symbols.

The first few nodes of a decision tree are represented in the schematic below.

.. figure:: extended_documentation/wxcode/thunder_nodes.png
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
forecast, proceed to the succeed node, else move to the fail node.

**Encoding a decision tree**

The first node above is encoded as follows::

  {
    "lightning": {
        "succeed": "lightning_cloud",
        "fail": "heavy_precipitation",
        "diagnostic_missing_action": "fail",
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
can be targeted as a succeed or fail destination from other nodes. The dictionary
accessed with this key contains the essentials that make the node function.

  - **succeed** (str or int): The next node to test if the condition in this
    node is true. Alternatively this may be an integer number that identifies
    which weather symbol has been reached; this is for the leaf (or final)
    nodes in the tree.
  - **fail** (str or int): The next node to test if the condition in this node
    is false. Alternatively this may be an integer number that identifies which
    weather symbol has been reached; this is for the leaf (or final) nodes in
    the tree.
  - **diagnostic_missing_action** (str(bool), optional): If the expected
    diagnostic is not provided, should the tree proceed to the succeed or fail
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
  - **diagnostic_thresholds** (List(List(float, str)): The diagnostic threshold
    value and units being used in the test. A threshold [value, units] pair must
    be provided for each diagnostic field with the same nested list structure; as
    the basic unit is a list of value and unit, the overall nested structure is
    one list deeper.
  - **diagnostic_conditions** (as diagnostic_fields): The expected inequality
    that has been used to construct the input probability field. This is checked
    against the spp__relative_to_threshold attribute of the threshold coordinate
    in the provided diagnostic.

Every decision tree must have a starting node, and this is taken as the first
node defined in the dictionary.

Manipulation of the diagnostics is possible using the decision tree configuration
to enable more complex comparisons. For example::

  "heavy_rain_or_sleet_shower": {
      "succeed": 14,
      "fail": 17,
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
