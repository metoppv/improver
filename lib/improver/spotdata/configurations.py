"""
Test data extraction configuration

"""
from os import environ as Environ
DIAGNOSTIC_FILE_PATH = Environ.get('DIAGNOSTIC_FILE_PATH')


def all_diagnostics():
    '''
    Defines how all available diagnostics should be processed. A custom name
    used to key the returned dictionary allows for multiple variations on the
    derivation of any one diagnostic for different products.

    e.g.
    temperature - might use intelligent grid point neighbour finding to help
                  near the coasts, and use a model_level_temperature_lapse_rate
                  to adjust the extracted data for unresolved topography.

    temperature_simple - might use simple nearest neighbour finding and
                         use_nearest data extraction to simply take the value
                         from that neighbouring point.

    '''
    diagnostic_recipes = {
        'temperature': {
            'filepath': (DIAGNOSTIC_FILE_PATH + '/*/*' +
                         'temperature_at_screen_level' + '*'),
            'diagnostic_name': 'air_temperature',
            'neighbour_finding': 'default',
            'interpolation_method': 'use_nearest',
            'extrema': True
            },
        'temperature_orog': {
            'filepath': (DIAGNOSTIC_FILE_PATH + '/*/*' +
                         'temperature_at_screen_level' + '*'),
            'diagnostic_name': 'air_temperature',
            'neighbour_finding': 'default',
            'interpolation_method': 'orography_derived_temperature_lapse_rate',
            'extrema': True
            },
        'wind_speed': {
            'filepath': (DIAGNOSTIC_FILE_PATH + '/*/*' +
                         'horizontal_wind_speed_and_direction_at_10m' + '*'),
            'diagnostic_name': 'wind_speed',
            'neighbour_finding': 'default',
            'interpolation_method': 'use_nearest',
            'extrema': False
            },
        'wind_direction': {
            'filepath': (DIAGNOSTIC_FILE_PATH + '/*/*' +
                         'horizontal_wind_speed_and_direction_at_10m' + '*'),
            'diagnostic_name': 'wind_from_direction',
            'neighbour_finding': 'default',
            'interpolation_method': 'use_nearest',
            'extrema': False
            },
        'visibility': {
            'filepath': (DIAGNOSTIC_FILE_PATH + '/*/*' +
                         'visibility_at_screen_level' + '*'),
            'diagnostic_name': 'visibility_in_air',
            'neighbour_finding': 'default',
            'interpolation_method': 'use_nearest',
            'extrema': True
            },
        'relative_humidity': {
            'filepath': (DIAGNOSTIC_FILE_PATH + '/*/*' +
                         'relative_humidity_at_screen_level' + '*'),
            'diagnostic_name': 'relative_humidity',
            'neighbour_finding': 'default',
            'interpolation_method': 'use_nearest',
            'extrema': False
            },
        'surface_pressure': {
            'filepath': (DIAGNOSTIC_FILE_PATH + '/*/*' +
                         'surface_pressure' + '*'),
            'diagnostic_name': 'surface_air_pressure',
            'neighbour_finding': 'default',
            'interpolation_method': 'use_nearest',
            'extrema': False
            },
        'low_cloud_amount': {
            'filepath': (DIAGNOSTIC_FILE_PATH + '/*/*' +
                         'low_cloud_amount' + '*'),
            'diagnostic_name': 'low_type_cloud_area_fraction',
            'neighbour_finding': 'default',
            'interpolation_method': 'use_nearest',
            'extrema': False
            },
        'medium_cloud_amount': {
            'filepath': (DIAGNOSTIC_FILE_PATH + '/*/*' +
                         'medium_cloud_amount' + '*'),
            'diagnostic_name': 'medium_type_cloud_area_fraction',
            'neighbour_finding': 'default',
            'interpolation_method': 'use_nearest',
            'extrema': False
            },
        'high_cloud_amount': {
            'filepath': (DIAGNOSTIC_FILE_PATH + '/*/*' +
                         'high_cloud_amount' + '*'),
            'diagnostic_name': 'high_type_cloud_area_fraction',
            'neighbour_finding': 'default',
            'interpolation_method': 'use_nearest',
            'extrema': False
            },
        'total_cloud_amount': {
            'filepath': DIAGNOSTIC_FILE_PATH + '/*/*' + 'total_cloud' + '*',
            'diagnostic_name': 'cloud_area_fraction',
            'neighbour_finding': 'default',
            'interpolation_method': 'use_nearest',
            'extrema': False
            }
        }
    return diagnostic_recipes


def define_diagnostics(configuration):
    '''
    Define the configurations with which spotdata may be run. These
    configurations specify which diagnostic definitions to include
    when processing for a given product.

    The routine also defines a default method of grid point neighbour
    finding diagnostics configurations that simply refer to 'default'.

    Args:
    -----
    configuration : A string used as a key in the configuration dictionary
                    to select the configuration for use.

    Returns:
    --------
    Dictionary containing the diagnostics to be processed and the definition
    of how to process them.

    '''
    diagnostics = all_diagnostics()

    configuration_dict = {
        'pws_default':
            ['temperature', 'wind_speed', 'wind_direction', 'visibility',
             'relative_humidity', 'surface_pressure', 'low_cloud_amount',
             'medium_cloud_amount', 'high_cloud_amount', 'total_cloud_amount'],

        'short_test':
            ['temperature', 'wind_speed']
        }

    neighbour_finding_default = {
        'pws_default': 'fast_nearest_neighbour',
        'short_test': 'fast_nearest_neighbour'
        }

    return (neighbour_finding_default[configuration],
            dict((key, diagnostics[key])
                 for key in configuration_dict[configuration]))
