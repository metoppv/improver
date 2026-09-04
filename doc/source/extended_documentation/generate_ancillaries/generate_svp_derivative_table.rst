**First derivative of the saturated vapour pressure (SVP)**

Output is a table of values for the first derivative of the SVP with respect to temperature.

The first derivative SVP is provided for the input temperature values, for which default values
are provided, ranging from 183.15 K to 338.15 K (inclusive) in 0.1 K increments. This results in
a table of 1550 rows (data points).

The resulting table of SVP derivative values has been plotted against the input temperature values in
the figure below. Here we can see a positive rate of change of the SVP with respect to temperature,
which is expected, as the SVP increases with increasing temperature. We can also see a kink in the line
around 273.16 K, which is the triple point of water where the three phases (solid, liquid, and gas)
coexist. This kink is caused by different equations being used for the SVP below and above this temperature.

.. figure:: extended_documentation/generate_ancillaries/saturated_vapour_pressure_derivative.png
     :align: center
     :scale: 80 %
     :alt: The first derivative of the saturated vapour pressure (SVP) with respect to temperature.     
