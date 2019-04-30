**Examples**

For a cube containing coefficients calculated using Ensemble
Model Output Statistics::

 emos_coefficients / (1)             (coefficient_index: 4)
     Dimension coordinates:
          coefficient_index                           x
     Auxiliary coordinates:
          coefficient_name                            x
     Scalar coordinates:
          forecast_period: 14400 seconds
          forecast_reference_time: 2017-11-10 00:00:00
          time: 2017-11-10 04:00:00
     Attributes:
          diagnostic_standard_name: air_temperature
          mosg__model_configuration: uk_det


An example of the coefficient_index coordinate is::

 DimCoord(array([0, 1, 2, 3]), standard_name=None, units=Unit('1'), long_name='coefficient_index')

An example of the coefficient_name coordinate is::

 AuxCoord(array(['gamma', 'delta', 'alpha', 'beta'], dtype='<U5'), standard_name=None, units=Unit('no_unit'), long_name='coefficient_name')

