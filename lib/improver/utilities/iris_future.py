import iris

IRIS_MAJOR_VERSION = float(iris.__version__.split(".")[0])


def set_future(property_name, value):
    if property_name in ("netcdf_promote", "netcdf_no_unlimited",
                         "cell_datetime_objects"):
        if IRIS_MAJOR_VERSION < 2:
            setattr(iris.FUTURE, property_name, value)
