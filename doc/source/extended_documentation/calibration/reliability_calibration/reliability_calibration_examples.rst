**Examples**

The reliability calibration tables returned by this plugin are structured as shown below::

  reliability_calibration_table / (1) (air_temperature: 2; table_row_index: 3; probability_bin: 5; projection_y_coordinate: 970; projection_x_coordinate: 1042)
       Dimension coordinates:
            air_temperature                           x                   -                   -                           -                             -
            table_row_index                           -                   x                   -                           -                             -
            probability_bin                           -                   -                   x                           -                             -
            projection_y_coordinate                   -                   -                   -                           x                             -
            projection_x_coordinate                   -                   -                   -                           -                             x
       Auxiliary coordinates:
            table_row_name                            -                   x                   -                           -                             -
       Scalar coordinates:
            cycle_hour: 22
            forecast_period: 68400 seconds
       Attributes:
            institution: Met Office
            source: Met Office Unified Model
            title: Reliability calibration data table

The leading dimension is threshold (here shown with air_temperature thresholds).
The reliability index dimension corresponds to the expected reliability table
rows, where these are observation count, sum of forecast probability, and forecast
count. The probability bins column will have a size that corresponds to the user
provided n_probability_bins. The last two dimensions are the spatial coordinates
on which the forecast and truth data has been provided.
