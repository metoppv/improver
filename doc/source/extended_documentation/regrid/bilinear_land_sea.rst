################################################
Bilinear Regridding with Land-sea Mask Awareness
################################################

This subpackage provides a fast and unified implementation of bilinear regridding with land-sea mask awareness. It is assumed that the source grid is evenly-spaced in a rectilinear latitude/longitude system. The target grid could be rectilinear or curvilinear. Regridding a field with land-sea mask awareness means that only land-surface source points are used for regridding at the land-surface target points, and vice versa at the sea-surface target points.

*******************************
Calculation Steps of Regridding
*******************************
 
* Given a target point, determine its four source points according to regular grid point ordering.

* Identify surface type of source/target points from land sea mask input files.

* Identify if source points have the same surface type as the target point. 
  The matched source points are called "True", and unmatched source points are called "False". 

* Calculate weights of four source points with normal bilinear formulation.

* Adjust source points & weights if unmatched source points exist. 

* Apply weights at values of source points for interpolated value in a unified way
  so that vectorisation of the Numpy computation can be fully achieved.
  
*********************************
Weight Calculation and Adjustment
*********************************

::
                      
                2   .----------. 3
                    |          |       target point x(lon,lat)
                lat |    x     |       four source points 1,2,3 and 4
                    |          |
                1   .----------. 4
                        lon
 

1. For the target points with four matched surrounding source points, their source point
   weights can be  calculated using the Finite-Element based interpolation shape function (please see 
   `R.D.Cook et al., 2001`_ on how to derive the finite element shape function). The formulations are
   as below (*var* is the physical variable to be interpolated).
   
.. _R.D.Cook et al., 2001: https://www.wiley.com/en-au/exportProduct/pdf/9780471356059

    .. math:: 
            var &= \sum \limits_{i=1}^4 weight_i \cdot var_i

            weight_1 &= (lat_2-lat)*(lon_3-lon)/area

            weight_2 &= (lat-lat_1)*(lon_3-lon)/area

            weight_3 &= (lat-lat_1)*(lon-lon_1)/area

            weight_4 &= (lat_2-lat)*(lon-lon_1)/area

            area &= (lat_2-lat_1)*(lon_3-lon_1)
        
     

2.  For the target points with three matched source points plus one unmatached source point, and
    the target point inside the triangle formed with three matched source points, three matched 
    source points is used to construct linear field for interpolation. The Finite-Element based
    interpolation shape functions are given for the weight calculation.

  - if source point 1 is unmatched:
    
        .. math::
            weight_1 &= 0.0

            weight_2 &= (lat_2-lat_1)*(lon_3-lon)/area

            weight_3 &= (lat_2-lat)*(lon_3-lon_1)/area

            weight_4 &= 1.0-weight_2-weight_3

  - if source point 2 is unmatched:  
   
        .. math::  
            weight_1 &= (lat_2-lat_1)*(lon_3-lon)/area

            weight_2 &= 0.0

            weight_3 &= (lat-lat_1)*(lon_3-lon_1)/area

            weight_4 &= 1.0-weight_1-weight_3

  - if source point 3 is unmatched:  

    .. math::  
        weight_2 &= (lat-lat_1)*(lon_3-lon_1)/area

        weight_4 &= (lat_2-lat_1)*(lon-lon_1)/area

        weight_3 &= 0.0

        weight_1 &= 1.0-weight_2-weight_4
    
  - if source point 4 is unmatched:
   
    .. math::  
        weight_1 &= (lat_2-lat)*(lon_3-lon_1)/area

        weight_3 &= (lat_2-lat_1)*(lon-lon_1)/area

        weight_2 &= 1.0-weight_1-weight_3

        weight_4 &= 0.0
    
    
3. The inverse distance weighting(IDW) method is used for the target points with the following:

  - One unmatched source points, three matched source points but the target point is outside
    the triangle formed with three matched source points. 
  - Two unmatched source points, two matched source points.
  - Three unmatched source points, one matched source point

The source points of inverse distance weighting are obtained by searching for the nearest k
(default: 4) source points for each target point using the K-D tree method, and then are identified
as unmatched-surface-type and matched-surface-type. If there is no matched source points, these
target points will be added into the target point list with “four unmatched source points”. The latitude and longitude coordinates are transformed to the earth-centric earth-fixed(ECEF) Cartesian XYZ coordinates for the K-D tree calculation so that the distances are calculated in the ECEF coordinate system. 

If there are matched surface points, the weights of inverse distance weighting are calculated: 

.. math::  
    weight_i &= \frac{w_i}{\sum \limits_{i=1}^N w_i}
    
    w_i &= \frac{1}{distance_i^p}    
    
where p is a positive number, called the power parameter. The optimum value =1.80 is used.

4. For the target points with four unmatched source points and zero matched source point, 
   re-locating their source points by looking up eight nearest source points with specified
   distance limit using the K-D tree method and then check if there are any same-type source points: 
  
  - if yes, use the matched source for the interpolation of inverse distance weighting  
  - if no, just ignore the surface type and do normal bilinear interpolation

