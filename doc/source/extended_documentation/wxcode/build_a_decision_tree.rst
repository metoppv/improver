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

  - **succeed**: The next node to test if the tests in this node succeed.
    Alternatively this may be an integer number that identifies which weather
    symbol has been reached; this is for the leaf (or final) nodes in the tree.
  - **fail**: The next node to test if the tests in this node fail.
    Alternatively this may be an integer number that identifies which weather
    symbol has been reached; this is for the leaf (or final) nodes in the tree.
  - **diagnostic_missing_action** (optional): If the expected diagnostic is not
    provided, should the tree proceed to the fail or succeed node. This can be
    useful if the tree is to be applied to output from different models, some of
    which do not provide all the diagnostics that might be desirable.
  - **probability_thresholds**: The probability that must be exceeded or not
    exceeded (see threshold_condition) for the node to progress to the succeed
    target.
  - **threshold_condition**: Defines the inequality test to be applied to the
    probability threshold. Inequalities that can be used are "<=", "<", ">", ">=".
  - **condition_combination**: If multiple tests are being applied in a single
    node, this value determines the logic with which they are combined. The
    values can be "AND", "OR".
  - **diagnostic_fields**: The name of the diagnostic that is to be used in the
    tests in this node. There may be multiple diagnostics if they are being
    combined in the test using a condition_combination, or if they are being
    manipulated within the node (e.g. added together).
  - **diagnostic_thresholds**: The diagnostic threshold value being used in the
    test. A threshold [value, units] pair must be provided for each diagnostic
    field.
  - **diagnostic_conditions**: The expected inequality that has been used to
    construct the input probability field. This is checked against the
    spp__relative_to_threshold attribute of the threshold coordinate in the
    provided diagnostic.
