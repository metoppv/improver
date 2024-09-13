# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
# flake8: noqa
"""
This module contains the plugins for the IMPROVER project.  This aids in discoverability
by making them available to a single flat namespace.  This also protects end-users from
changes in structure to IMPROVER impacting their use of the plugins.
"""
from importlib import import_module

# alphabetically sorted IMPROVER plugin lookup
PROCESSING_MODULES = {
    "Accumulation": "improver.nowcasting.accumulation",
    "AdjustLandSeaPoints": "improver.regrid.landsea",
    "AdvectField": "improver.nowcasting.forecasting",
    "AggregateReliabilityCalibrationTables": "improver.calibration.reliability_calibration",
    "apply_mask": "improver.utilities.mask",
    "ApplyBiasCorrection": "improver.calibration.simple_bias_correction",
    "ApplyDecisionTree": "improver.categorical.decision_tree",
    "ApplyDzRescaling": "improver.calibration.dz_rescaling",
    "ApplyEMOS": "improver.calibration.ensemble_calibration",
    "ApplyGriddedLapseRate": "improver.lapse_rate",
    "ApplyNeighbourhoodProcessingWithAMask": "improver.nbhood.use_nbhood",
    "ApplyOrographicEnhancement": "improver.nowcasting.utilities",
    "ApplyRainForestsCalibration": "improver.calibration.rainforest_calibration",
    "ApplyReliabilityCalibration": "improver.calibration.reliability_calibration",
    "BaseNeighbourhoodProcessing": "improver.nbhood.nbhood",
    "CalculateForecastBias": "improver.calibration.simple_bias_correction",
    "CalibratedForecastDistributionParameters": "improver.calibration.ensemble_calibration",
    "ChooseDefaultWeightsLinear": "improver.blending.weights",
    "ChooseDefaultWeightsNonLinear": "improver.blending.weights",
    "ChooseDefaultWeightsTriangular": "improver.blending.weights",
    "ChooseWeightsLinear": "improver.blending.weights",
    "CloudCondensationLevel": "improver.psychrometric_calculations.cloud_condensation_level",
    "CloudTopTemperature": "improver.psychrometric_calculations.cloud_top_temperature",
    "Combine": "improver.cube_combiner",
    "ConstructReliabilityCalibrationTables": "improver.calibration.reliability_calibration",
    "ContinuousRankedProbabilityScoreMinimisers": "improver.calibration.ensemble_calibration",
    "ConvectionRatioFromComponents": "improver.precipitation_type.convection",
    "ConvertProbabilitiesToPercentiles": "improver.ensemble_copula_coupling.ensemble_copula_coupling",
    "CopyAttributes": "improver.utilities.copy_attributes",
    "CorrectLandSeaMask": "improver.generate_ancillaries.generate_ancillary",
    "CreateExtrapolationForecast": "improver.nowcasting.forecasting",
    "CubeCombiner": "improver.cube_combiner",
    "DayNightMask": "improver.utilities.solar",
    "DifferenceBetweenAdjacentGridSquares": "improver.utilities.spatial",
    "EnforceConsistentForecasts": "improver.utilities.forecast_reference_enforcement",
    "EnsembleReordering": "improver.ensemble_copula_coupling.ensemble_copula_coupling",
    "EstimateCoefficientsForEnsembleCalibration": "improver.calibration.ensemble_calibration",
    "EstimateDzRescaling": "improver.calibration.dz_rescaling",
    "ExpectedValue": "improver.expected_value",
    "ExtendRadarMask": "improver.nowcasting.utilities",
    "ExtractLevel": "improver.utilities.cube_extraction",
    "ExtractSubCube": "improver.utilities.cube_extraction",
    "FieldTexture": "improver.utilities.textural",
    "FillRadarHoles": "improver.nowcasting.utilities",
    "FreezingRain": "improver.precipitation_type.freezing_rain",
    "FrictionVelocity": "improver.wind_calculations.wind_downscaling",
    "GenerateClearskySolarRadiation": "improver.generate_ancillaries.generate_derived_solar_fields",
    "GenerateOrographyBandAncils": "improver.generate_ancillaries.generate_ancillary",
    "GenerateSolarTime": "improver.generate_ancillaries.generate_derived_solar_fields",
    "GenerateTimeLaggedEnsemble": "improver.utilities.time_lagging",
    "GenerateTopographicZoneWeights": "improver.generate_ancillaries.generate_topographic_zone_weights",
    "GradientBetweenAdjacentGridSquares": "improver.utilities.spatial",
    "HailFraction": "improver.precipitation_type.hail_fraction",
    "HailSize": "improver.psychrometric_calculations.hail_size",
    "HumidityMixingRatio": "improver.psychrometric_calculations.psychrometric_calculations",
    "Integration": "improver.utilities.mathematical_operations",
    "InterpolateUsingDifference": "improver.utilities.interpolation",
    "LapseRate": "improver.lapse_rate",
    "LightningFromCapePrecip": "improver.lightning",
    "LightningMultivariateProbability_USAF2024": "improver.lightning",
    "ManipulateReliabilityTable": "improver.calibration.reliability_calibration",
    "MaxInTimeWindow": "improver.cube_combiner",
    "MergeCubes": "improver.utilities.cube_manipulation",
    "MergeCubesForWeightedBlending": "improver.blending.weighted_blend",
    "MetaCloudCondensationLevel": "improver.psychrometric_calculations.cloud_condensation_level",
    "MetaNeighbourhood": "improver.nbhood.nbhood",
    "MetaWetBulbFreezingLevel": "improver.psychrometric_calculations.wet_bulb_temperature",
    "ModalCategory": "improver.categorical.modal_code",
    "NeighbourSelection": "improver.spotdata.neighbour_finding",
    "NowcastLightning": "improver.nowcasting.lightning",
    "OccurrenceBetweenThresholds": "improver.between_thresholds",
    "OccurrenceWithinVicinity": "improver.utilities.spatial",
    "OpticalFlow": "improver.nowcasting.optical_flow",
    "OrographicEnhancement": "improver.orographic_enhancement",
    "OrographicSmoothingCoefficients": "improver.generate_ancillaries.generate_orographic_smoothing_coefficients",
    "PercentileConverter": "improver.percentile",
    "PhaseChangeLevel": "improver.psychrometric_calculations.psychrometric_calculations",
    "PostProcessingPlugin": "improver.__init__",
    "PrecipPhaseProbability": "improver.psychrometric_calculations.precip_phase_probability",
    "PystepsExtrapolate": "improver.nowcasting.pysteps_advection",
    "RebadgePercentilesAsRealizations": "improver.ensemble_copula_coupling.ensemble_copula_coupling",
    "RebadgeRealizationsAsPercentiles": "improver.ensemble_copula_coupling.ensemble_copula_coupling",
    "RecursiveFilter": "improver.nbhood.recursive_filter",
    "RegridLandSea": "improver.regrid.landsea",
    "RegridWithLandSeaMask": "improver.regrid.landsea2",
    "ResamplePercentiles": "improver.ensemble_copula_coupling.ensemble_copula_coupling",
    "ResolveWindComponents": "improver.wind_calculations.wind_components",
    "RoughnessCorrection": "improver.wind_calculations.wind_downscaling",
    "SaturatedVapourPressureTable": "improver.generate_ancillaries.generate_svp_table",
    "ShowerConditionProbability": "improver.precipitation_type.shower_condition_probability",
    "SignificantPhaseMask": "improver.psychrometric_calculations.significant_phase_mask",
    "SnowFraction": "improver.precipitation_type.snow_fraction",
    "SnowSplitter": "improver.precipitation_type.snow_splitter",
    "SpatiallyVaryingWeightsFromMask": "improver.blending.spatial_weights",
    "SpotExtraction": "improver.spotdata.spot_extraction",
    "SpotHeightAdjustment": "improver.spotdata.height_adjustment",
    "SpotLapseRateAdjust": "improver.spotdata.apply_lapse_rate",
    "SpotManipulation": "improver.spotdata.spot_manipulation",
    "StandardiseMetadata": "improver.standardise",
    "TemporalInterpolation": "improver.utilities.temporal_interpolation",
    "Threshold": "improver.threshold",
    "TriangularWeightedBlendAcrossAdjacentPoints": "improver.blending.blend_across_adjacent_points",
    "VerticalUpdraught": "improver.wind_calculations.vertical_updraught",
    "VisibilityCombineCloudBase": "improver.visibility.visibility_combine_cloud_base",
    "WeightAndBlend": "improver.blending.calculate_weights_and_blend",
    "WeightedBlendAcrossWholeDimension": "improver.blending.weighted_blend",
    "WetBulbTemperature": "improver.psychrometric_calculations.wet_bulb_temperature",
    "WetBulbTemperatureIntegral": "improver.psychrometric_calculations.wet_bulb_temperature",
    "WindDirection": "improver.wind_calculations.wind_direction",
    "WindGustDiagnostic": "improver.wind_calculations.wind_gust_diagnostic",
}


def __getattr__(name):
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(f"{name} is not a valid attribute")
    mod = import_module(PROCESSING_MODULES[name])
    return getattr(mod, name)