import numpy as np
from iris.cube import Cube
from iris.analysis import Linear
from iris.coords import DimCoord


example_data = np.array([[2, 2],[-1, 1],[3, 3]])
dims = [(DimCoord([0, 10, 20], standard_name="latitude", units="degrees"), 0), (DimCoord([5, 15], standard_name="longitude", units="degrees"), 1)]

cube = Cube(example_data, dim_coords_and_dims=dims)

newdims_cube = Cube(np.zeros((3,3)), dim_coords_and_dims=[(DimCoord([0, 10, 20], standard_name="latitude", units="degrees"), 0), (DimCoord([0, 10, 20], standard_name="longitude", units="degrees"), 1)])

regridded = cube.regrid(newdims_cube, Linear())
print(regridded.data)