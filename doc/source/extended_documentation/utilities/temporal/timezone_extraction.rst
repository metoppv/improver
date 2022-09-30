**Local timezone forecasts**

For some diagnostics it makes sense for the forecasts to be expressed in
the local timezone of each location. For example, when calculating the
maximum day time temperature this will occur between different times in
different places across the globe. One approach to provide a suitable
forecast for all locations is to produce 12-hour maximum temperatures
every hour and then allow the user to select the correct UTC offset for
their location. This pushes complexity down stream of the IMPROVER
implementation and may often be desirable. However it also increases the
volume of data that must be served to users.

An alternative is to construct gridded or spot forecast outputs that are
valid for a given calendar day. These are produced by mapping forecasts
valid at the correct time for a given grid point or site to that grid
point or site in a single forecast output. This is achieved using the
TimezoneExtraction plugin. It takes as input an ancillary that masks
grid points (shown below) or sites in all UTC offsets other than the
one into which they fall.

.. figure:: extended_documentation/utilities/temporal/
    uk_timezones.png
    :align: center
    :scale: 100 %
    :alt: UK area timezone masks.

The plugin also takes a target local time as input, which is used to ensure
that the correct forecasts are provided. The final input to the plugin is
forecasts valid at times that span the UTC offset range required by the
ancillary and which are valid at the target local times.

The forecast data is stacked to align the time coordinate with the UTC offset
coordinate in the ancillary and the mask is then applied. Following this step
there will be only one forecast value that is not masked for each grid point
or site. The time coordinate can then be collapsed, keeping only unmasked
forecast values. This produces a forecast valid at the same local time in
each location on a given calendar day.

**Missing timezones**

For period diagnostics, such as maximum daytime temperature, the whole
period is required for the diagnostic to make sense. When a local timezone
product is constructed this requires that all the timezones to be populated
have a complete period input available. On a global domain this has a
significant impact on forecasts in the west. As soon as the earliest lead-time
in the forecast being produced reaches the next calendar day in the east of the
domain it is no longer possible to update the current calendar day local-
timezone forecast. In the west there may still be more up-to-date forecasts to
be produced for the period but these do not get incorporated. This effectively
means that western regions always receive forecasts at longer lead-times which
will be of poorer quality.

As such the plugin must be able handle entirely missing timezones, producing
masked data in these regions to indicate that no forecast information is
available. (YET TO BE IMPLEMENTED).

**Partial periods**

For diagnostics that are designed to reflect the conditions on a given day
and which are updated throughout a given day the plugin must handle partial
periods. Such diagnostics might be a daily weather symbol. This symbol may
be updated throughout the current day (on which forecasts are being produced)
to reflect the conditions that remain to be seen in the day, rather than being
dominated by weather conditions that have passed.

The plugin is able to handle these cases. As the resulting product includes
a time coordinate point and bounds for every forecast location this results
in inconsistent bounds across all locations, but they are recorded to ensure
the data is described.