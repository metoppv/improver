#######################################
Ingestion of DataFrames into iris cubes
#######################################

DataFrames of the forecasts and truths (observations) can be provided
for use with Ensemble Model Output Statistics (EMOS). The format
expected for the forecast and truths DataFrames is described below.
The forecasts are ensemble site forecasts in percentile format at
a set of observation sites. The truths are observations from
observation sites.

****************************
Forecast DataFrame
****************************

The forecast DataFrame is expected to contain the following compulsory
columns: forecast, blend_time, forecast_period, forecast_reference_time,
time, wmo_id, percentile, diagnostic, latitude, longitude, period, height,
cf_name, units and experiment. Other columns will be ignored.

A summary of the expected contents of a forecast table is shown below.

.. csv-table::
    :file: ./forecast_dataframe_metadata_info.csv
    :widths: 25, 22, 53
    :header-rows: 1

An example forecast table for an instantaneous diagnostic is shown below.

.. csv-table::
    :file: ./forecast_dataframe_instantaneous_example.csv
    :header-rows: 1

An example forecast table for a period diagnostic is shown below.

.. csv-table::
    :file: ./forecast_dataframe_period_example.csv
    :header-rows: 1

****************************
Truth DataFrame
****************************

The truth DataFrame is expected to contain the following compulsory
columns: ob_value, time, wmo_id, diagnostic, latitude, longitude and
altitude. Other columns will be ignored.

A summary of the expected contents of a truth table is shown below.

.. csv-table::
    :file: ./truth_dataframe_metadata_info.csv
    :widths: 30, 30, 40
    :header-rows: 1

An example truth table is shown below.

.. csv-table::
    :file: ./truth_dataframe_example.csv
    :header-rows: 1
