# (C) Crown copyright, Met Office. All rights reserved.
#
# This file is part of IMPROVER and is released under a BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""
This module contains the plugins for the IMPROVER project.  This aids in discoverability
by making them available to a single flat namespace.  This also protects end-users from
changes in structure to IMPROVER impacting their use of the plugins.
"""
from improver.between_thresholds import OccurrenceBetweenThresholds
from improver.blending.blend_across_adjacent_points import (
    TriangularWeightedBlendAcrossAdjacentPoints,
)
from improver.blending.calculate_weights_and_blend import WeightAndBlend
from improver.blending.spatial_weights import SpatiallyVaryingWeightsFromMask
from improver.blending.weighted_blend import (
    MergeCubesForWeightedBlending,
    WeightedBlendAcrossWholeDimension,
)
from improver.blending.weights import (
    ChooseDefaultWeightsLinear,
    ChooseDefaultWeightsNonLinear,
    ChooseDefaultWeightsTriangular,
    ChooseWeightsLinear,
)
from improver.calibration.dz_rescaling import ApplyDzRescaling, EstimateDzRescaling
from improver.calibration.ensemble_calibration import (
    ApplyEMOS,
    CalibratedForecastDistributionParameters,
    ContinuousRankedProbabilityScoreMinimisers,
    EstimateCoefficientsForEnsembleCalibration,
)
from improver.calibration.rainforest_calibration import ApplyRainForestsCalibration
from improver.calibration.reliability_calibration import (
    AggregateReliabilityCalibrationTables,
    ApplyReliabilityCalibration,
    ConstructReliabilityCalibrationTables,
    ManipulateReliabilityTable,
)
from improver.calibration.simple_bias_correction import (
    ApplyBiasCorrection,
    CalculateForecastBias,
)
from improver.categorical.decision_tree import ApplyDecisionTree
from improver.categorical.modal_code import ModalCategory
from improver.cube_combiner import Combine, CubeCombiner, MaxInTimeWindow
from improver.ensemble_copula_coupling.ensemble_copula_coupling import (
    ConvertProbabilitiesToPercentiles,
    EnsembleReordering,
    RebadgePercentilesAsRealizations,
    RebadgeRealizationsAsPercentiles,
    ResamplePercentiles,
)
from improver.expected_value import ExpectedValue
from improver.generate_ancillaries.generate_ancillary import (
    CorrectLandSeaMask,
    GenerateOrographyBandAncils,
)
from improver.generate_ancillaries.generate_derived_solar_fields import (
    GenerateClearskySolarRadiation,
    GenerateSolarTime,
)
from improver.generate_ancillaries.generate_orographic_smoothing_coefficients import (
    OrographicSmoothingCoefficients,
)
from improver.generate_ancillaries.generate_svp_table import (
    SaturatedVapourPressureTable,
)
from improver.generate_ancillaries.generate_topographic_zone_weights import (
    GenerateTopographicZoneWeights,
)
from improver.lapse_rate import ApplyGriddedLapseRate, LapseRate
from improver.lightning import (
    LightningFromCapePrecip,
    LightningMultivariateProbability_USAF2024,
)
from improver.nbhood.nbhood import BaseNeighbourhoodProcessing, MetaNeighbourhood
from improver.nbhood.recursive_filter import RecursiveFilter
from improver.nbhood.use_nbhood import ApplyNeighbourhoodProcessingWithAMask
from improver.nowcasting.accumulation import Accumulation
from improver.nowcasting.forecasting import AdvectField, CreateExtrapolationForecast
from improver.nowcasting.lightning import NowcastLightning
from improver.nowcasting.optical_flow import OpticalFlow
from improver.nowcasting.pysteps_advection import PystepsExtrapolate
from improver.nowcasting.utilities import (
    ApplyOrographicEnhancement,
    ExtendRadarMask,
    FillRadarHoles,
)
from improver.orographic_enhancement import OrographicEnhancement
from improver.percentile import PercentileConverter
from improver.precipitation_type.convection import ConvectionRatioFromComponents
from improver.precipitation_type.freezing_rain import FreezingRain
from improver.precipitation_type.hail_fraction import HailFraction
from improver.precipitation_type.shower_condition_probability import (
    ShowerConditionProbability,
)
from improver.precipitation_type.snow_fraction import SnowFraction
from improver.precipitation_type.snow_splitter import SnowSplitter
from improver.psychrometric_calculations.cloud_condensation_level import (
    CloudCondensationLevel,
    MetaCloudCondensationLevel,
)
from improver.psychrometric_calculations.cloud_top_temperature import (
    CloudTopTemperature,
)
from improver.psychrometric_calculations.hail_size import HailSize
from improver.psychrometric_calculations.precip_phase_probability import (
    PrecipPhaseProbability,
)
from improver.psychrometric_calculations.psychrometric_calculations import (
    HumidityMixingRatio,
    PhaseChangeLevel,
)
from improver.psychrometric_calculations.significant_phase_mask import (
    SignificantPhaseMask,
)
from improver.psychrometric_calculations.wet_bulb_temperature import (
    MetaWetBulbFreezingLevel,
    WetBulbTemperature,
    WetBulbTemperatureIntegral,
)
from improver.regrid.landsea import AdjustLandSeaPoints, RegridLandSea
from improver.regrid.landsea2 import RegridWithLandSeaMask
from improver.spotdata.apply_lapse_rate import SpotLapseRateAdjust
from improver.spotdata.height_adjustment import SpotHeightAdjustment
from improver.spotdata.neighbour_finding import NeighbourSelection
from improver.spotdata.spot_extraction import SpotExtraction
from improver.standardise import StandardiseMetadata
from improver.threshold import Threshold
from improver.utilities.copy_attributes import CopyAttributes
from improver.utilities.cube_extraction import ExtractLevel, ExtractSubCube
from improver.utilities.cube_manipulation import MergeCubes
from improver.utilities.forecast_reference_enforcement import EnforceConsistentForecasts
from improver.utilities.interpolation import InterpolateUsingDifference
from improver.utilities.mathematical_operations import Integration
from improver.utilities.solar import DayNightMask
from improver.utilities.spatial import (
    DifferenceBetweenAdjacentGridSquares,
    GradientBetweenAdjacentGridSquares,
    OccurrenceWithinVicinity,
)
from improver.utilities.temporal_interpolation import TemporalInterpolation
from improver.utilities.textural import FieldTexture
from improver.utilities.time_lagging import GenerateTimeLaggedEnsemble
from improver.visibility.visibility_combine_cloud_base import VisibilityCombineCloudBase
from improver.wind_calculations.vertical_updraught import VerticalUpdraught
from improver.wind_calculations.wind_components import ResolveWindComponents
from improver.wind_calculations.wind_direction import WindDirection
from improver.wind_calculations.wind_downscaling import (
    FrictionVelocity,
    RoughnessCorrection,
)
from improver.wind_calculations.wind_gust_diagnostic import WindGustDiagnostic

__all__ = [
    "Accumulation",
    "AdjustLandSeaPoints",
    "AdvectField",
    "AggregateReliabilityCalibrationTables",
    "ApplyBiasCorrection",
    "ApplyDecisionTree",
    "ApplyDzRescaling",
    "ApplyEMOS",
    "ApplyGriddedLapseRate",
    "ApplyNeighbourhoodProcessingWithAMask",
    "ApplyOrographicEnhancement",
    "ApplyRainForestsCalibration",
    "ApplyReliabilityCalibration",
    "BaseNeighbourhoodProcessing",
    "CalculateForecastBias",
    "CalibratedForecastDistributionParameters",
    "ChooseDefaultWeightsLinear",
    "ChooseDefaultWeightsNonLinear",
    "ChooseDefaultWeightsTriangular",
    "ChooseWeightsLinear",
    "CloudCondensationLevel",
    "CloudTopTemperature",
    "Combine",
    "ConstructReliabilityCalibrationTables",
    "ContinuousRankedProbabilityScoreMinimisers",
    "ConvectionRatioFromComponents",
    "ConvertProbabilitiesToPercentiles",
    "CopyAttributes",
    "CorrectLandSeaMask",
    "CreateExtrapolationForecast",
    "CubeCombiner",
    "DayNightMask",
    "DifferenceBetweenAdjacentGridSquares",
    "EnforceConsistentForecasts",
    "EnsembleReordering",
    "EstimateCoefficientsForEnsembleCalibration",
    "EstimateDzRescaling",
    "ExpectedValue",
    "ExtendRadarMask",
    "ExtractLevel",
    "ExtractSubCube",
    "FieldTexture",
    "FillRadarHoles",
    "FreezingRain",
    "FrictionVelocity",
    "GenerateClearskySolarRadiation",
    "GenerateOrographyBandAncils",
    "GenerateSolarTime",
    "GenerateTimeLaggedEnsemble",
    "GenerateTopographicZoneWeights",
    "GradientBetweenAdjacentGridSquares",
    "HailFraction",
    "HailSize",
    "HumidityMixingRatio",
    "Integration",
    "InterpolateUsingDifference",
    "LapseRate",
    "LightningFromCapePrecip",
    "LightningMultivariateProbability_USAF2024",
    "ManipulateReliabilityTable",
    "MaxInTimeWindow",
    "MergeCubes",
    "MergeCubesForWeightedBlending",
    "MetaCloudCondensationLevel",
    "MetaNeighbourhood",
    "MetaWetBulbFreezingLevel",
    "ModalCategory",
    "NeighbourSelection",
    "NowcastLightning",
    "OccurrenceBetweenThresholds",
    "OccurrenceWithinVicinity",
    "OpticalFlow",
    "OrographicEnhancement",
    "OrographicSmoothingCoefficients",
    "PercentileConverter",
    "PhaseChangeLevel",
    "PrecipPhaseProbability",
    "PystepsExtrapolate",
    "RebadgePercentilesAsRealizations",
    "RebadgeRealizationsAsPercentiles",
    "RecursiveFilter",
    "RegridLandSea",
    "RegridWithLandSeaMask",
    "ResamplePercentiles",
    "ResolveWindComponents",
    "RoughnessCorrection",
    "SaturatedVapourPressureTable",
    "ShowerConditionProbability",
    "SignificantPhaseMask",
    "SnowFraction",
    "SnowSplitter",
    "SpatiallyVaryingWeightsFromMask",
    "SpotExtraction",
    "SpotHeightAdjustment",
    "SpotLapseRateAdjust",
    "StandardiseMetadata",
    "TemporalInterpolation",
    "Threshold",
    "TriangularWeightedBlendAcrossAdjacentPoints",
    "VerticalUpdraught",
    "VisibilityCombineCloudBase",
    "WeightAndBlend",
    "WeightedBlendAcrossWholeDimension",
    "WetBulbTemperature",
    "WetBulbTemperatureIntegral",
    "WindDirection",
    "WindGustDiagnostic",
]
