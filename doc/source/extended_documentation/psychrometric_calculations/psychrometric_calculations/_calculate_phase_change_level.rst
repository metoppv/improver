**Method**

The first call is to ``find_falling_level`` which finds the height
at which the wet-bulb integral reaches the required threshold, this
threshold defining the phase change level. The threshold value depends
upon the phase change that has been requested. If we are unable to find
the height of the phase level for a grid point from the wet-bulb
integral the value at that point is set to np.nan and will be filled
in later.

The next call is to ``fill_in_high_phase_change_falling_levels``.
This function is used to fill in any data in the phase change level
where this level is very high, effectively above the highest level
used in the wet bulb temperature integration. In these areas the
atmosphere is so warm that the phase change has occured above the
highest level that is being considered. In these areas the phase
change level is set to the highest height level + the height of the
orography. The exact height of the phase change level is unimportant
in these areas as we can be certain the precipitation at the surface
will be rain.

Next we fill in any sea points where we have not found a phase change
level by the time we get to sea level, i.e. where the wet bulb
temperature integral never reaches the desired threshold. To fill
these points we call ``fill_in_sea_points`` which finds a linear fit to
the wet bulb temperature close to sea level and uses this to find where
an extrapolated wet bulb temperature integral would cross the threshold.
This results in phase change levels below sea level for points where we
have applied the extrapolation. It is important to get sensible values
for these heights, particularly in marginal snow fall cases. As we
calculate probabilities using multiple realizations, if some
realizations give a snow falling level just above sea level and some
just below, getting these heights close to correct ensures we calculate
realistic probabilities of snow falling at the surface. Consider instead
if we simply set the falling levels at these points to be equal to sea
level; in so doing we would be introducing a warm bias to the output.

Finally, if there are any areas for which the phase change level remains
unset, we know that these must be at or below the height of the
orography. As such we fill these in using horizontal interpolation across
the grid. This is achieved using two calls to
``interpolate_missing_data``. The first attempts to use linear
interpolation to fill as many of these holes as possible. The second
uses nearest neighbour filling for any areas where linear interpolation
was not possible, e.g. at the edge of the domain. The process is as
follows:

  1. Fill in the phase change level for points with no value yet
     set using horizontal interpolation from surrounding set points.
     Only interpolate from surrounding set points at which the phase
     change level is below the maximum orography height in the region
     around the unset point. This helps us avoid spreading very high
     phase change levels across areas where we had missing data.
  2. Fill any gaps that still remain where the linear interpolation has
     not been able to find a value because there is not enough
     data (e.g at the corners of the domain). Use nearest neighbour
     filling.
  3. Check whether despite our efforts we have still filled in some
     of the missing points with phase change levels above the orography.
     In these cases set the missing points to the height of orography.

If there are any points that have somehow remain unset, these are filled
with the missing_data value.
