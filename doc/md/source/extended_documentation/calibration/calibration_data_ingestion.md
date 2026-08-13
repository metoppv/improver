# Ingestion of DataFrames into iris cubes

DataFrames of the forecasts and truths (observations) can be provided
for use with Ensemble Model Output Statistics (EMOS). The format
expected for the forecast and truths DataFrames is described below. The
forecasts are ensemble site forecasts in percentile format at a set of
observation sites. The truths are observations from observation sites.

## Forecast DataFrame

The forecast DataFrame is expected to contain the following compulsory
columns: forecast; blend_time; forecast_period; forecast_reference_time;
time; wmo_id; diagnostic; latitude; longitude; period; height; cf_name;
units; experiment; and exactly one of percentile or realization.
Optionally, the DataFrame may also contain station_id. If the truth
DataFrame also contains station_id, then forecast and truth data will be
matched using both wmo_id and station_id. The station_id data may be
either string or int. Any other columns not mentioned above will be
ignored.

A summary of the expected contents of a forecast table is shown below.

  --------------------------------------------------------------------------------------
  Column                    Dtype                  Notes
  ------------------------- ---------------------- -------------------------------------
  forecast                  float64                The value for a particular forecast.

  altitude                  float32                The altitude in metres.

  blend_time                datetime64\[ns,UTC\]   The time at which a blend of models
                                                   was produced.

  forecast_period           timedelta64\[ns\]      The difference between the blend time
                                                   (and forecast reference time) and the
                                                   validity time.

  forecast_reference_time   datetime64\[ns,UTC\]   The time at which the forecast
                                                   analysis was made for a forecast from
                                                   a single source. Equal to the
                                                   blend_time for a forecast created
                                                   from blending multiple forecast
                                                   sources.

  latitude                  float32                The latitude in degrees.

  longitude                 float32                The longitude in degrees.

  time                      datetime64\[ns,UTC\]   The validity time of the forecasts.
                                                   Signifies the end of the forecast
                                                   period for period diagnostics.

  wmo_id                    object                 The five digit WMO ID.

  station_id                object or int          Optional additional site identifier.

  cf_name                   object                 The CF name for the diagnostic. From
                                                   DataFrames consisting of one
                                                   diagnostic this is expected to be
                                                   constant.

  units                     object                 The units of the forecast value. From
                                                   DataFrames consisting of one
                                                   diagnostic this is expected to be
                                                   constant.

  percentile                float64                The percentile value.

  realization               int                    The realization number.

  period                    timedelta64\[ns\]      The period the forecast valid is
                                                   over. Set to missing data for
                                                   instantaneous forecasts.

  height                    float32                The height of the forecast value.
                                                   From DataFrames consisting of one
                                                   diagnostic this is expected to be
                                                   constant.

  diagnostic                category               The name of the diagnostic. From
                                                   DataFrames consisting of one
                                                   diagnostic this is expected to be
                                                   constant.

  experiment                object                 A value used for identifying how the
                                                   data was generated when the table
                                                   contains multiple equivalent
                                                   forecasts.
  --------------------------------------------------------------------------------------

An example forecast table for an instantaneous diagnostic is shown
below.

<table>
<thead>
<tr>
<th>index</th>
<th>forecast</th>
<th>altitude</th>
<th>blend_time</th>
<th>forecast_period</th>
<th>forecast_reference_time</th>
<th>latitude</th>
<th>longitude</th>
<th>time</th>
<th>wmo_id</th>
<th>cf_name</th>
<th>units</th>
<th>percentile</th>
<th>period</th>
<th>height</th>
<th>diagnostic</th>
<th>experiment</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>282.69</td>
<td>15</td>
<td>2021-08-01 18:00:00+00:00</td>
<td>1 days</td>
<td>2021-08-01 18:00:00+00:00</td>
<td>60</td>
<td><dl>
<dt><code>-5</code></dt>
<dd>
&#10;</dd>
</dl></td>
<td>2021-08-02 18:00:00+00:00</td>
<td>03001</td>
<td>air_temperature</td>
<td>K</td>
<td>5</td>
<td>NaT</td>
<td>1.5</td>
<td>temperature_at_screen_level</td>
<td>threshold</td>
</tr>
<tr>
<td>1</td>
<td>283.2</td>
<td>82</td>
<td>2021-08-01 18:00:00+00:00</td>
<td>1 days</td>
<td>2021-08-01 18:00:00+00:00</td>
<td>59</td>
<td><dl>
<dt><code>-4</code></dt>
<dd>
&#10;</dd>
</dl></td>
<td>2021-08-02 18:00:00+00:00</td>
<td>03002</td>
<td>air_temperature</td>
<td>K</td>
<td>5</td>
<td>NaT</td>
<td>1.5</td>
<td>temperature_at_screen_level</td>
<td>threshold</td>
</tr>
<tr>
<td>2</td>
<td>282.62</td>
<td>30</td>
<td>2021-08-01 18:00:00+00:00</td>
<td>1 days</td>
<td>2021-08-01 18:00:00+00:00</td>
<td>58</td>
<td><dl>
<dt><code>-3</code></dt>
<dd>
&#10;</dd>
</dl></td>
<td>2021-08-02 18:00:00+00:00</td>
<td>03003</td>
<td>air_temperature</td>
<td>K</td>
<td>5</td>
<td>NaT</td>
<td>1.5</td>
<td>temperature_at_screen_level</td>
<td>threshold</td>
</tr>
<tr>
<td>3</td>
<td>286.17</td>
<td>4</td>
<td>2021-08-01 18:00:00+00:00</td>
<td>1 days</td>
<td>2021-08-01 18:00:00+00:00</td>
<td>57</td>
<td><dl>
<dt><code>-2</code></dt>
<dd>
&#10;</dd>
</dl></td>
<td>2021-08-02 18:00:00+00:00</td>
<td>03004</td>
<td>air_temperature</td>
<td>K</td>
<td>5</td>
<td>NaT</td>
<td>1.5</td>
<td>temperature_at_screen_level</td>
<td>threshold</td>
</tr>
<tr>
<td>4</td>
<td>284.43</td>
<td>15</td>
<td>2021-08-01 18:00:00+00:00</td>
<td>1 days</td>
<td>2021-08-01 18:00:00+00:00</td>
<td>56</td>
<td><dl>
<dt><code>-1</code></dt>
<dd>
&#10;</dd>
</dl></td>
<td>2021-08-02 18:00:00+00:00</td>
<td>03005</td>
<td>air_temperature</td>
<td>K</td>
<td>5</td>
<td>NaT</td>
<td>1.5</td>
<td>temperature_at_screen_level</td>
<td>threshold</td>
</tr>
</tbody>
</table>

An example forecast table for an instantaneous diagnostic including
station_id is shown below. The last 3 rows will be represented as
different spot_index values in the output, since they have different
station_id.

<table>
<thead>
<tr>
<th>index</th>
<th>forecast</th>
<th>altitude</th>
<th>blend_time</th>
<th>forecast_period</th>
<th>forecast_reference_time</th>
<th>latitude</th>
<th>longitude</th>
<th>time</th>
<th>wmo_id</th>
<th>station_id</th>
<th>cf_name</th>
<th>units</th>
<th>percentile</th>
<th>period</th>
<th>height</th>
<th>diagnostic</th>
<th>experiment</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>282.69</td>
<td>15</td>
<td>2021-08-01 18:00:00+00:00</td>
<td>1 days</td>
<td>2021-08-01 18:00:00+00:00</td>
<td>60</td>
<td><dl>
<dt><code>-5</code></dt>
<dd>
&#10;</dd>
</dl></td>
<td>2021-08-02 18:00:00+00:00</td>
<td>03001</td>
<td>029233</td>
<td>air_temperature</td>
<td>K</td>
<td>5</td>
<td>NaT</td>
<td>1.5</td>
<td>temperature_at_screen_level</td>
<td>threshold</td>
</tr>
<tr>
<td>1</td>
<td>283.2</td>
<td>82</td>
<td>2021-08-01 18:00:00+00:00</td>
<td>1 days</td>
<td>2021-08-01 18:00:00+00:00</td>
<td>59</td>
<td><dl>
<dt><code>-4</code></dt>
<dd>
&#10;</dd>
</dl></td>
<td>2021-08-02 18:00:00+00:00</td>
<td>03002</td>
<td>029234</td>
<td>air_temperature</td>
<td>K</td>
<td>5</td>
<td>NaT</td>
<td>1.5</td>
<td>temperature_at_screen_level</td>
<td>threshold</td>
</tr>
<tr>
<td>2</td>
<td>282.62</td>
<td>30</td>
<td>2021-08-01 18:00:00+00:00</td>
<td>1 days</td>
<td>2021-08-01 18:00:00+00:00</td>
<td>58</td>
<td><dl>
<dt><code>-3</code></dt>
<dd>
&#10;</dd>
</dl></td>
<td>2021-08-02 18:00:00+00:00</td>
<td>00000</td>
<td>029235</td>
<td>air_temperature</td>
<td>K</td>
<td>5</td>
<td>NaT</td>
<td>1.5</td>
<td>temperature_at_screen_level</td>
<td>threshold</td>
</tr>
<tr>
<td>3</td>
<td>286.17</td>
<td>4</td>
<td>2021-08-01 18:00:00+00:00</td>
<td>1 days</td>
<td>2021-08-01 18:00:00+00:00</td>
<td>57</td>
<td><dl>
<dt><code>-2</code></dt>
<dd>
&#10;</dd>
</dl></td>
<td>2021-08-02 18:00:00+00:00</td>
<td>00000</td>
<td>029236</td>
<td>air_temperature</td>
<td>K</td>
<td>5</td>
<td>NaT</td>
<td>1.5</td>
<td>temperature_at_screen_level</td>
<td>threshold</td>
</tr>
<tr>
<td>4</td>
<td>284.43</td>
<td>15</td>
<td>2021-08-01 18:00:00+00:00</td>
<td>1 days</td>
<td>2021-08-01 18:00:00+00:00</td>
<td>56</td>
<td><dl>
<dt><code>-1</code></dt>
<dd>
&#10;</dd>
</dl></td>
<td>2021-08-02 18:00:00+00:00</td>
<td>00000</td>
<td>029237</td>
<td>air_temperature</td>
<td>K</td>
<td>5</td>
<td>NaT</td>
<td>1.5</td>
<td>temperature_at_screen_level</td>
<td>threshold</td>
</tr>
</tbody>
</table>

An example forecast table for a period diagnostic is shown below.

<table>
<thead>
<tr>
<th>index</th>
<th>forecast</th>
<th>altitude</th>
<th>blend_time</th>
<th>forecast_period</th>
<th>forecast_reference_time</th>
<th>latitude</th>
<th>longitude</th>
<th>time</th>
<th>wmo_id</th>
<th>cf_name</th>
<th>units</th>
<th>percentile</th>
<th>period</th>
<th>height</th>
<th>diagnostic</th>
<th>experiment</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>282.69</td>
<td>15</td>
<td>2021-08-01 00:00:00+00:00</td>
<td>0 days 09:00:00</td>
<td>2021-08-01 00:00:00+00:00</td>
<td>60</td>
<td><dl>
<dt><code>-5</code></dt>
<dd>
&#10;</dd>
</dl></td>
<td>2021-08-01 21:00:00+00:00</td>
<td>03001</td>
<td>temperature_at_screen_level_daytime_max</td>
<td>K</td>
<td>5</td>
<td>0 days 12:00:00</td>
<td>1.5</td>
<td>temperature_at_screen_level_max-daytime</td>
<td>threshold</td>
</tr>
<tr>
<td>1</td>
<td>283.2</td>
<td>82</td>
<td>2021-08-01 00:00:00+00:00</td>
<td>0 days 09:00:00</td>
<td>2021-08-01 00:00:00+00:00</td>
<td>59</td>
<td><dl>
<dt><code>-4</code></dt>
<dd>
&#10;</dd>
</dl></td>
<td>2021-08-01 21:00:00+00:00</td>
<td>03002</td>
<td>temperature_at_screen_level_daytime_max</td>
<td>K</td>
<td>5</td>
<td>0 days 12:00:00</td>
<td>1.5</td>
<td>temperature_at_screen_level_max-daytime</td>
<td>threshold</td>
</tr>
<tr>
<td>2</td>
<td>282.62</td>
<td>30</td>
<td>2021-08-01 00:00:00+00:00</td>
<td>0 days 09:00:00</td>
<td>2021-08-01 00:00:00+00:00</td>
<td>58</td>
<td><dl>
<dt><code>-3</code></dt>
<dd>
&#10;</dd>
</dl></td>
<td>2021-08-01 21:00:00+00:00</td>
<td>03003</td>
<td>temperature_at_screen_level_daytime_max</td>
<td>K</td>
<td>5</td>
<td>0 days 12:00:00</td>
<td>1.5</td>
<td>temperature_at_screen_level_max-daytime</td>
<td>threshold</td>
</tr>
<tr>
<td>3</td>
<td>286.17</td>
<td>4</td>
<td>2021-08-01 00:00:00+00:00</td>
<td>0 days 09:00:00</td>
<td>2021-08-01 00:00:00+00:00</td>
<td>57</td>
<td><dl>
<dt><code>-2</code></dt>
<dd>
&#10;</dd>
</dl></td>
<td>2021-08-01 21:00:00+00:00</td>
<td>03004</td>
<td>temperature_at_screen_level_daytime_max</td>
<td>K</td>
<td>5</td>
<td>0 days 12:00:00</td>
<td>1.5</td>
<td>temperature_at_screen_level_max-daytime</td>
<td>threshold</td>
</tr>
<tr>
<td>4</td>
<td>284.43</td>
<td>15</td>
<td>2021-08-01 00:00:00+00:00</td>
<td>0 days 09:00:00</td>
<td>2021-08-01 00:00:00+00:00</td>
<td>56</td>
<td><dl>
<dt><code>-1</code></dt>
<dd>
&#10;</dd>
</dl></td>
<td>2021-08-01 21:00:00+00:00</td>
<td>03005</td>
<td>temperature_at_screen_level_daytime_max</td>
<td>K</td>
<td>5</td>
<td>0 days 12:00:00</td>
<td>1.5</td>
<td>temperature_at_screen_level_max-daytime</td>
<td>threshold</td>
</tr>
</tbody>
</table>

An example forecast table for a period diagnostic including station_id
is shown below.

<table>
<thead>
<tr>
<th>index</th>
<th>forecast</th>
<th>altitude</th>
<th>blend_time</th>
<th>forecast_period</th>
<th>forecast_reference_time</th>
<th>latitude</th>
<th>longitude</th>
<th>time</th>
<th>wmo_id</th>
<th>station_id</th>
<th>cf_name</th>
<th>units</th>
<th>percentile</th>
<th>period</th>
<th>height</th>
<th>diagnostic</th>
<th>experiment</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>282.69</td>
<td>15</td>
<td>2021-08-01 00:00:00+00:00</td>
<td>0 days 09:00:00</td>
<td>2021-08-01 00:00:00+00:00</td>
<td>60</td>
<td><dl>
<dt><code>-5</code></dt>
<dd>
&#10;</dd>
</dl></td>
<td>2021-08-01 21:00:00+00:00</td>
<td>03001</td>
<td>029233</td>
<td>temperature_at_screen_level_daytime_max</td>
<td>K</td>
<td>5</td>
<td>0 days 12:00:00</td>
<td>1.5</td>
<td>temperature_at_screen_level_max-daytime</td>
<td>threshold</td>
</tr>
<tr>
<td>1</td>
<td>283.2</td>
<td>82</td>
<td>2021-08-01 00:00:00+00:00</td>
<td>0 days 09:00:00</td>
<td>2021-08-01 00:00:00+00:00</td>
<td>59</td>
<td><dl>
<dt><code>-4</code></dt>
<dd>
&#10;</dd>
</dl></td>
<td>2021-08-01 21:00:00+00:00</td>
<td>03002</td>
<td>029234</td>
<td>temperature_at_screen_level_daytime_max</td>
<td>K</td>
<td>5</td>
<td>0 days 12:00:00</td>
<td>1.5</td>
<td>temperature_at_screen_level_max-daytime</td>
<td>threshold</td>
</tr>
<tr>
<td>2</td>
<td>282.62</td>
<td>30</td>
<td>2021-08-01 00:00:00+00:00</td>
<td>0 days 09:00:00</td>
<td>2021-08-01 00:00:00+00:00</td>
<td>58</td>
<td><dl>
<dt><code>-3</code></dt>
<dd>
&#10;</dd>
</dl></td>
<td>2021-08-01 21:00:00+00:00</td>
<td>00000</td>
<td>029235</td>
<td>temperature_at_screen_level_daytime_max</td>
<td>K</td>
<td>5</td>
<td>0 days 12:00:00</td>
<td>1.5</td>
<td>temperature_at_screen_level_max-daytime</td>
<td>threshold</td>
</tr>
<tr>
<td>3</td>
<td>286.17</td>
<td>4</td>
<td>2021-08-01 00:00:00+00:00</td>
<td>0 days 09:00:00</td>
<td>2021-08-01 00:00:00+00:00</td>
<td>57</td>
<td><dl>
<dt><code>-2</code></dt>
<dd>
&#10;</dd>
</dl></td>
<td>2021-08-01 21:00:00+00:00</td>
<td>00000</td>
<td>029236</td>
<td>temperature_at_screen_level_daytime_max</td>
<td>K</td>
<td>5</td>
<td>0 days 12:00:00</td>
<td>1.5</td>
<td>temperature_at_screen_level_max-daytime</td>
<td>threshold</td>
</tr>
<tr>
<td>4</td>
<td>284.43</td>
<td>15</td>
<td>2021-08-01 00:00:00+00:00</td>
<td>0 days 09:00:00</td>
<td>2021-08-01 00:00:00+00:00</td>
<td>56</td>
<td><dl>
<dt><code>-1</code></dt>
<dd>
&#10;</dd>
</dl></td>
<td>2021-08-01 21:00:00+00:00</td>
<td>00000</td>
<td>029237</td>
<td>temperature_at_screen_level_daytime_max</td>
<td>K</td>
<td>5</td>
<td>0 days 12:00:00</td>
<td>1.5</td>
<td>temperature_at_screen_level_max-daytime</td>
<td>threshold</td>
</tr>
</tbody>
</table>

## Truth DataFrame

The truth DataFrame is expected to contain the following compulsory
columns: ob_value, time, wmo_id, diagnostic, latitude, longitude and
altitude. Optionally, the DataFrame may also contain station_id and
units. If the forecast DataFrame also contains station_id, then forecast
and truth data will be matched using both wmo_id and station_id. Other
columns will be ignored. If the truth DataFrame contains a units column,
then it will be used for the units of the output truth cube. Otherwise,
the units of the truth cube will be copied from the units of the
forecast DataFrame. The station_id data may be either string or int. Any
other columns not mentioned above will be ignored.

A summary of the expected contents of a truth table is shown below.

  -------------------------------------------------------------------------
  Column                Dtype                  Notes
  --------------------- ---------------------- ----------------------------
  time                  datetime64\[ns,UTC\]   The time of the observation.

  wmo_id                object                 The five digit WMO ID.

  latitude              float32                The latitude in degrees.

  longitude             float32                The longtitude in degrees.

  altitude              float32                The altitude in metres.

  ob_value              float32                The value for a particular
                                               observation.

  diagnostic            category               The name of the diagnostic.

  units                 str                    Optional units of the
                                               observation values.
  -------------------------------------------------------------------------

An example truth table is shown below.

+-------+----------+----------+----------+-----------+----------------+--------+-----------------------------+
| index | ob_value | altitude | latitude | longitude | time           | wmo_id | diagnostic                  |
+=======+==========+==========+==========+===========+================+========+=============================+
| 0     | 283.45   | 15       | 60       | `-5`      | 2021-08-02     | 03001  | temperature_at_screen_level |
|       |          |          |          |           | 18:00:00+00:00 |        |                             |
|       |          |          |          | :         |                |        |                             |
+-------+----------+----------+----------+-----------+----------------+--------+-----------------------------+
| 1     | 283.91   | 82       | 59       | `-4`      | 2021-08-02     | 03002  | temperature_at_screen_level |
|       |          |          |          |           | 18:00:00+00:00 |        |                             |
|       |          |          |          | :         |                |        |                             |
+-------+----------+----------+----------+-----------+----------------+--------+-----------------------------+
| 2     | 281.63   | 30       | 58       | `-3`      | 2021-08-02     | 03003  | temperature_at_screen_level |
|       |          |          |          |           | 18:00:00+00:00 |        |                             |
|       |          |          |          | :         |                |        |                             |
+-------+----------+----------+----------+-----------+----------------+--------+-----------------------------+
| 3     | 286.55   | 4        | 57       | `-2`      | 2021-08-02     | 03004  | temperature_at_screen_level |
|       |          |          |          |           | 18:00:00+00:00 |        |                             |
|       |          |          |          | :         |                |        |                             |
+-------+----------+----------+----------+-----------+----------------+--------+-----------------------------+
| 4     | 283.19   | 15       | 56       | `-1`      | 2021-08-02     | 03005  | temperature_at_screen_level |
|       |          |          |          |           | 18:00:00+00:00 |        |                             |
|       |          |          |          | :         |                |        |                             |
+-------+----------+----------+----------+-----------+----------------+--------+-----------------------------+
