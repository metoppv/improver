Construction of the precipitation duration diagnostics involves a lot of input data. When working with a large ensemble, and constructing a long period from many short periods, the array sizes can become very memory hungry if operated on all at once. Every additional threshold that is included in the processing further exacibates this issue. For example, input cubes something like this:

```
realizations = 50
times = 8 x 3-hour periods to construct a 24-hour target.
rate / accumulation thresholds = 3
y/x = 1000
```

means we are combining and collapsing two cubes of shape (50, 8, 3, 1000, 1000).

We work around this using slicing, breaking the data up into smaller chunks that can be read into memory one at a time, their contribution assessed, and then the data freed. Still, to generate percentiles in a standard manner we would still need to hold all of the processed data in memory together and collapse across the realization coordinate. How data hungry this is depends upon the data.

The method used in this plugin is as follows:

1. Extract the requested rate and accumulation threshold cubes from the input cubes.
2. Combine the extracted cubes to form a single max rate and a single accumlation cube, each with a time dimension, as well a thresold dimension if multiple thresholds have been requested.
3. If we are working with scalar (single valued) threshold coordinates, escalate these to dimensions anyway so we can work with arrays with a consistent number of dimensions.
4. Arrange the dimensions into a specific order so we know that when we index these we are working with the expected dimensions.
5. Generate a tuple of tuples that describe all the different combinations of rate and accumulation thresholds to be combined together to classify the precipitation.
6. Scale the percentiles requested by the user to values in the range 0 to (realizations - 1).
7. Create a target array into which data will be placed. Note that if the number of percentiles being generated were as large as the number of realizations, we would not be doing much to reduce the memory with this approach.
8. Slice the data over the realization coordinate. We are therefore working with arrays with dimensions (threshold, time, y, x).
9. Realize the data by touching it, `cube.data`. This is done to reduce the amount of lazy loading of the data when we further subset it. Without this step the lazy loading and cleaning up of data takes much longer, making the plugin far too slow.
10. Loop over the different combinations of rate and accumulation thresholds. The inputs show be binary probabilities, so we multiply them together to determine if a given location at a given time may be classified as precipitating by exceeding the rate and accumulation thresholds.
11. We sum over the time coordinate to get the total number of periods at each location that have been classified as precipitating.
12. As we are working with discrete periods we know all the values (number of periods) that could be returned by the sum over time. We use an array that has as one of its dimensions a length equal to this set of potential values. For each threshold combination and location we then add 1 into an array at an index in this dimension that corresponds to the value (number of periods) that was calculated. For example if a 24-hour period is being constructed from 3-hour period inputs, the possible values are between 0 (no periods classified) and 8 (all periods classified).
13. Once we have looped over all realizations we have a `hit_count` array of shape (accumulation_thresholds, rate_thresholds, N, y, x) where `N` corresponds to the possible number of periods. This is effectively a frequency table, telling us how many realizations, for each location, classified 0, 1, 2 etc. periods as precipitating. We loop again over the rate and accumulation thresholds and sum cumulatively along the periods dimension. The sum that's returned is therefore a count of how many realizations classified each location as having fewer than or equal to `n` many periods classified as precipitating.
14. We can find the index in this cumulative list at which the various requested percentiles would fall. This allows us to get to the percentile we are after without having all of the values in `hit_count` represented individually in some array. The lower and upper bounds are found, i.e. our target percentile value sits between these two values.
15. We perform a linear interpolation between the values to obtain our target percentile value.
16. The calcualted percentiles are stored for each combination of rate and accumulation thresholds and location, with these then written out to an output cube with suitable metadata.