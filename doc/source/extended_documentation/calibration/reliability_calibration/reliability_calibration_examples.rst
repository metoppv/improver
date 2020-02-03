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
The table row index dimension corresponds to the expected reliability table
rows. These rows are named in the associated table_row_name coordinate, with the
names being observation_count, sum_of_forecast_probability, and forecast_count.
The probability bins column will have a size that corresponds to the user
provided n_probability_bins. The last two dimensions are the spatial coordinates
on which the forecast and truth data has been provided.

Note that the probability bins are defined as a series of non-overlapping ranges.
Adjacent bin boundaries are spaced at the smallest representable interval, such
that no probability can fall outside of any bin. The probability bins may include
single value limits if the user chooses, where these are bins with a width of
1.0E-6, at 0 (0 to 1.0E-6) and 1 ((1 - 1.0E-6) to 1). These finite widths ensure
that float precision errors do not prevent values being allocated to these bins.
The equality operators away from these single limit bins are effectively "greater
than" the lower bound of the probability bin, and "less than or equal to" the
upper bound.
