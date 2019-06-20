**Examples**

Included here are some examples of where changing data precision can have
a significant impact on the data.

**Time**

When changing the data type of time coordinates it is important to ensure
that the loss of decimals does not fundamentally alter the time that is
recorded. Consider a time::

  2018-11-03 16:15:00 expressed in seconds since 1970-01-01 00:00:00
  1541261700 seconds

If the units are changed to hours since 1970-01-01 00:00:00 this becomes::

  2018-11-03 16:15:00 expressed in seconds since 1970-01-01 00:00:00
  428128.25 hours

If as well as changing units, the data type is changed from float to integer
it becomes obvious that the fractional hour component will be lost. This
results in the time finally becoming::

  2018-11-03 16:00:00 expressed in seconds since 1970-01-01 00:00:00
  428128 hours

Thus in the process of compelling a cube to conform to default units and
data types (hours and integers respectively) we have altered the data to
a significant extent.


**Spatial**

Imagine a spatial coordinate defined at 1000m intervals, running from 0 to
5000m. This coordinate could readily be changed to km and these could be
stored as integers without anything fundamentally changing::

  0. 1000. 2000. 3000. 4000. 5000. m
  0 1 2 3 4 5 km

However, if the coordinates are offset by 500m from the origin this is no
longer true::

  500. 1500. 2500. 3500. 4500. 5500. m
  0 1 2 3 4 5 km

Once again it is important to check that the change of data type will not
result in a loss of significant information.
