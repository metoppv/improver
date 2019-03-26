**Examples**

The neighbour_cube required for use in this method is constructed using the :func:`~improver.spotdata_new.neighbour_finding.NeighbourSelection` class. Such a cube will contain grid point neighbours to spot sites found by one or more neighbour_selection_methods. The vertical displacement between the spot data site and the grid point neighbour is the third attribute shown in the cube below. The spot site information is also contained within the cube, stored within auxiliary coordinates. An example of such a cube is shown here, where in this case grid point neighbours are provided for 4 different methods of choosing them::

  grid_neighbours / (1)               (spot_index: 121; neighbour_selection_method: 4; grid_attributes: 3)
       Dimension coordinates:
            spot_index                           x                                -                   -
            neighbour_selection_method           -                                x                   -
            grid_attributes                      -                                -                   x
       Auxiliary coordinates:
            altitude                             x                                -                   -
            latitude                             x                                -                   -
            longitude                            x                                -                   -
            wmo_id                               x                                -                   -
            neighbour_selection_method_name      -                                x                   -
            grid_attributes_key                  -                                -                   x
       Attributes:
            Conventions: CF-1.5
            mosg__grid_domain: uk_extended
            mosg__grid_type: standard
            mosg__grid_version: 1.2.0
            mosg__model_configuration: uk_det


The diagnostic cube will be a standard IMPROVER diagnostic cube, where there may be dimensions leading the x and y spatial coordinates. These will be maintained in the output cube. So for an input diagnostic cube such as::

  air_temperature / (K)               (realization: 12; projection_y_coordinate: 970; projection_x_coordinate: 1042)
       Dimension coordinates:
            realization                           x                            -                             -
            projection_y_coordinate               -                            x                             -
            projection_x_coordinate               -                            -                             x
       Scalar coordinates:
            forecast_period: 0 seconds
            forecast_reference_time: 2018-07-23 09:00:00
            height: 1.5 m
            time: 2018-07-23 09:00:00
       Attributes:
            Conventions: CF-1.5
            history: 2018-07-23T12:57:06Z: StaGE Decoupler
            institution: Met Office
            mosg__grid_domain: uk_extended
            mosg__grid_type: standard
            mosg__grid_version: 1.2.0
            mosg__model_configuration: uk_ens
            source: Met Office Unified Model
            title: MOGREPS-UK Model Forecast on UK 2 km Standard Grid
            um_version: 10.8


The resulting output cube will have the following form, in which the spatial coordinates have been removed and in there place is the information about the spot data sites. Note that the data provenance metadata is also copied forward onto the output cube::

  air_temperature / (K)               (realization: 12; spot_index: 121)
       Dimension coordinates:
            realization                           x               -
            spot_index                            -               x
       Auxiliary coordinates:
            altitude                              -               x
            latitude                              -               x
            longitude                             -               x
            wmo_id                                -               x
       Scalar coordinates:
            forecast_period: 0 seconds
            forecast_reference_time: 2018-07-23 09:00:00
            height: 1.5 m
            time: 2018-07-23 09:00:00
       Attributes:
            Conventions: CF-1.5
            history: 2018-07-23T12:57:06Z: StaGE Decoupler
            institution: Met Office
            mosg__grid_domain: uk_extended
            mosg__grid_type: standard
            mosg__grid_version: 1.2.0
            mosg__model_configuration: uk_ens
            source: Met Office Unified Model
            title: MOGREPS-UK Model Forecast on UK 2 km Standard Grid
            um_version: 10.8
