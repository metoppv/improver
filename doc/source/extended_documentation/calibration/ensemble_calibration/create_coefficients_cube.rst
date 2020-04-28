**Examples**

For a cubelist containing coefficients calculated using Ensemble
Model Output Statistics::

 0: emos_coefficient_gamma / (K)        (scalar cube)
 1: emos_coefficient_beta / (1)         (scalar cube)
 2: emos_coefficient_alpha / (K)        (scalar cube)
 3: emos_coefficient_delta / (1)        (scalar cube)

An example cube is therefore::

 emos_coefficient_gamma / (K)        (scalar cube)
      Scalar coordinates:
           forecast_period: 43200 seconds
           forecast_reference_time: 2017-06-05 03:00:00
           projection_x_coordinate: -159000.0 m, bound=(-358000.0, 40000.0) m
           projection_y_coordinate: -437000.0 m, bound=(-636000.0, -238000.0) m
           time: 2017-06-05 15:00:00
      Attributes:
           Conventions: CF-1.5
           diagnostic_standard_name: air_temperature
           mosg__model_configuration: uk_ens
