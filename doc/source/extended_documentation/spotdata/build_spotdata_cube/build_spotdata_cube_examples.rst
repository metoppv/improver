**Examples**

For a cube containing diagnostic values::

 air_temperature / (K)             (index: 5)
     Dimension coordinates:
          time                            -
          index                           x
     Auxiliary coordinates:
          forecast_period                 -
          altitude                        x
          latitude                        x
          longitude                       x
          utc_offset                      x
          wmo_site                        x
     Scalar coordinates:
          forecast_reference_time: 2018-07-13 09:00:00
          height: 1.5 m

Or for a cube identifying grid point neighbours::

 ukvx_grid_neighbour / (1)           (spot_index: 100; neighbour_selection_method: 3; grid_attribute: 3)
     Dimension coordinates:
          spot_index                           x                                -                  -
          neighbour_selection_method           -                                x                  -
          grid_attribute                       -                                -                  x
     Auxiliary coordinates:
          altitude                             x                                -                  -
          latitude                             x                                -                  -
          longitude                            x                                -                  -
          wmo_ids                              x                                -                  -
          neighbour_selection_method_name      -                                x                  -
          grid_attribute_key                   -                                -                  x


where the neighbour_selection dimension coordinate is necessarily numeric,
but the string names of the methods can be found in the associated
neighbour_selection_method_name auxiliary coordinate, e.g.::

 AuxCoord(array(['nearest', 'nearest_land', 'nearest_land_minimumdz', 'bilinear'], dtype='<U22'),
    standard_name=None, units=Unit('1'), long_name='neighbour_selection_method_name')

The grid attribute coordinate contains the information necessary for
extracting and adjusting diagnostic data from a gridded field. The names
of the elements of this coordinate can be found in the grid_attribute_key
auxiliary coordinate, e.g.::

 AuxCoord(array(['x_index', 'y_index', 'vertical_displacement'], dtype='<U22'),
    standard_name=None, units=Unit('1'), long_name='grid_attribute_key')
