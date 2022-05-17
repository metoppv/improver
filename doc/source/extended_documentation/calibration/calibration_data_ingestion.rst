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
columns: forecast; blend_time; forecast_period; forecast_reference_time;
time; wmo_id; diagnostic; latitude; longitude; period; height;
cf_name; units; experiment; and exactly one of percentile or realization.
Optionally, the DataFrame may also contain station_id. If the
truth DataFrame also contains station_id, then forecast and truth data
will be matched using both wmo_id and station_id. The station_id data may be
either string or int. Any other columns not mentioned above will be ignored.

A summary of the expected contents of a forecast table is shown below.

.. csv-table::
    :file: ./forecast_dataframe_metadata_info.csv
    :widths: 25, 22, 53
    :header-rows: 1

An example forecast table for an instantaneous diagnostic is shown below.

.. csv-table::
    :file: ./forecast_dataframe_instantaneous_example.csv
    :header-rows: 1


An example forecast table for an instantaneous diagnostic including station_id
is shown below. The last 3 rows will be represented as different spot_index
values in the output, since they have different station_id.

.. csv-table::
    :file: ./forecast_dataframe_instantaneous_example_with_station_id.csv
    :header-rows: 1

An example forecast table for a period diagnostic is shown below.

.. csv-table::
    :file: ./forecast_dataframe_period_example.csv
    :header-rows: 1


An example forecast table for a period diagnostic including station_id is shown below.

.. csv-table::
    :file: ./forecast_dataframe_period_example_with_station_id.csv
    :header-rows: 1

****************************
Truth DataFrame
****************************

The truth DataFrame is expected to contain the following compulsory
columns: ob_value, time, wmo_id, diagnostic, latitude, longitude and
altitude. Optionally, the DataFrame may also contain station_id and units. 
If the forecast DataFrame also contains station_id, then forecast and truth data
will be matched using both wmo_id and station_id. Other columns will be 
ignored. If the truth DataFrame contains a units column, then it will be used 
for the units of the output truth cube. Otherwise, the units of the truth cube
will be copied from the units of the forecast DataFrame. The station_id data may be
either string or int. Any other columns not mentioned above will be ignored.

A summary of the expected contents of a truth table is shown below.

.. csv-table::
    :file: ./truth_dataframe_metadata_info.csv
    :widths: 30, 30, 40
    :header-rows: 1

An example truth table is shown below.

.. csv-table::
    :file: ./truth_dataframe_example.csv
    :header-rows: 1
