**Using grid_point_radius**

A grid_point_radius will work with any projection but the effective kernel
shape in real space may be irregular. An exagerated example is shown in
the figure below. A vicinity radius of 1 grid point leads to a 3x3 grid
point area, defined here on a very coarse lat-lon (PlateCaree) grid. This
has been reprojected to an equal areas grid to demonstrate the resulting
trapezoidal shape of the vicinity in "real space".

.. figure:: extended_documentation/utilities/spatial/
    vicinity_example.png
    :align: center
    :scale: 60 %
    :alt: Grid point vicinity reprojected from Lat-Lon grid to equal areas.

Users must be aware of this when choosing whether to use this plugin with
a non-equal areas projection. It could introduce biases into the resulting
data, e.g. related to latitude in the above example.

**Vicinity processing of regridded coarse grid data**

When applying vicinity processing to data sourced from a coarse grid some
care is needed. The particular case of concern is coarse grid data that has
been regridded to a finer mesh without any kind of downscaling. This means
that the data on the finer mesh remains blocky, revealing the grid scale of
the underlying data. In such cases, though a vicinity radius of order the
fine mesh scale could be applied, there is no additional information on this
scale; vicinity processing is effectively a convolution to a coarser grid,
and it cannot be used to convolve to what is effectively a finer grid.

As such, when vicinity processing this kind of regridded but coarse data, the
minimum vicinity radius that should be used is one which returns a vicinity
region on the scale of the underlying coarse grid.

    (vicinity radius)_min = underlying grid cell width / 2
