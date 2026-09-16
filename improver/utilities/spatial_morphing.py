# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Plugin for spatial morphing between forecast sources using Google FILM."""

import json
import warnings
from typing import Any

import iris
import numpy as np
from iris.cube import Cube, CubeList
from scipy.ndimage import maximum_filter, uniform_filter

from improver import BasePlugin
from improver.blending.utilities import remove_blend_time, remove_deprecation_warnings
from improver.clustering.realization_clustering import RealizationSelection
from improver.utilities.temporal import (
    reset_forecast_reference_time_and_period,
    validate_cycletime_format,
)
from improver.utilities.temporal_interpolation import (
    GoogleFilmInterpolation,
    _as_tuple_if_list,
)

# Canonical suppression-stage order used by configuration and execution:
# - weak_signal: Damps broad low-intensity excess above a source-weighted
#   reference.
# - convective: Restores locally concentrated shower-like peaks indicated by
#   source-neighbourhood structure.
# - upper_tail: Restores high-end intensity where convective signal exists but
#   the morphed upper tail is weaker than source-derived upper-tail expectations.
_SUPPRESSION_CANONICAL_STAGES: tuple[str, str, str] = (
    "weak_signal",
    "convective",
    "upper_tail",
)

# Suppression tuning defaults. Each entry is documented with its intended meaning
# and the expected numerical range for the parameter:
# - occurrence_threshold: Minimum wet value in the same units as the data; values
#   below this threshold are treated as dry for the suppression diagnostics.
# - quantile_for_centre: Quantile in [0, 1] used to define the central intensity
#   level used by suppression weighting.
# - width_fraction: Positive scaling factor controlling how broad the suppression
#   weighting is around the centre; typically > 0.
# - maximum_suppression: Non-negative strength limit in [0, 1] for the maximum
#   allowed suppression.
# - showery_neighbourhood_size: Positive odd integer neighbourhood size used to
#   detect showery structure; typically >= 1.
# - showery_weight_factor: Weight in [0, 1] applied to the showery component.
# - weakness_weight_factor: Weight in [0, 1] applied to the local weakness
#   component.
# - convective_neighbourhood_size: Positive odd integer neighbourhood size used
#   for convective feature detection; typically >= 1.
# - convective_gain: Gain factor in [0, 1] applied to the convective
#   contribution.
# - concentration_reference: Positive reference concentration value used when
#   scaling the suppression response; typically > 0.
# - concentration_scale: Positive scaling factor controlling sensitivity to
#   concentration; typically > 0.
# - upper_tail_quantile: Quantile in [0, 1] used to identify upper-tail values.
# - convective_mask_threshold: Threshold in [0, 1] used to activate the
#   convective mask.
# - maximum_intensity_scale: Positive scaling factor >= 1 used as the upper bound
#   on intensity-based scaling.
# - upper_tail_intensity_quantile: Quantile in [0, 1] used for upper-tail
#   intensity weighting.
# - intensity_weight_width_fraction: Positive width control for intensity
#   weighting; typically > 0.
# - sigmoid_clip_limit: Non-negative clip limit for the normalised sigmoid input
#   used to avoid numerical overflow/underflow in the logistic weighting
#   function; typically > 0.
_SUPPRESSION_DEFAULTS: dict[str, float | int] = {
    "occurrence_threshold": 0.03,
    "quantile_for_centre": 0.9,
    "width_fraction": 2.0,
    "maximum_suppression": 1.0,
    "showery_neighbourhood_size": 11,
    "showery_weight_factor": 0.75,
    "weakness_weight_factor": 0.25,
    "convective_neighbourhood_size": 25,
    "convective_gain": 0.9,
    "concentration_reference": 2.0,
    "concentration_scale": 5.0,
    "upper_tail_quantile": 0.95,
    "convective_mask_threshold": 0.5,
    "maximum_intensity_scale": 1.5,
    "upper_tail_intensity_quantile": 0.75,
    "intensity_weight_width_fraction": 0.25,
    "sigmoid_clip_limit": 20.0,
}


class SpatialMorphing(BasePlugin):
    """Spatially morph between forecast sources for a selected realization cluster.

    This plugin builds upon RealizationSelection to select realizations from multiple
    forecast sources according to cluster assignments, then applies spatial morphing
    using Google FILM to create seamless transitions between different source models.

    Unlike hard joins (RealizationSelection alone), this plugin produces spatially
    smooth blended forecasts where different sources contribute smoothly based on
    configured transition characteristics.

    Workflow:
    1. Split input cubes into forecast cubes and cluster cube (from
       RealizationClusterAndMatch).
    2. Validate that all forecast cubes have the same validity time and, if
       requested, update forecast_reference_time using the supplied cycletime.
    3. Parse the cluster mapping attributes to identify the selected source and
       realization for the requested cluster at the target forecast period.
    4. Check whether the mapped source/realization is available on the provided
       input cubes; if not, fall back to an alternative valid source for the same
       cluster.
    5. Diagnose the source and realization pair that should be used for any
       active transition at this forecast period, including the two source models
       and their weights for the blended result.
    6. Extract the selected realizations from each contributing source and apply
       the configured morphing backend (Google FILM by default) to generate a
       seamless blended output.
     7. Optionally apply one or more local suppression stages to the morphed
        result:
        weak_signal damps broad low-intensity excess above a source-weighted
        reference; convective restores locally concentrated shower-like peaks
        indicated by the source neighbourhood structure; upper_tail restores
        high-end intensity where convective signal exists but the morphed upper
        tail is weaker than source-derived upper-tail expectations.
    8. Finalise the output cube by cleaning metadata, setting the selected
       cluster as the realization coordinate, and recording expected/actual
       forecast contributor provenance.

    This plugin is designed to work with output from RealizationClusterAndMatch,
    providing a more direct spatial morphing alternative to the
    RealizationSelection to ForecastTrajectoryGapFiller pipeline.
    """

    def __init__(
        self,
        forecast_period: int,
        cluster_number: int,
        model_id_attr: str = "mosg__model_configuration",
        cycletime: str | None = None,
        selection_attr: str | None = None,
        selection_attr_value: str = "spatial_morphing",
        transitions: dict[str, Any] | None = None,
        model_path: str | None = None,
        scaling: str = "minmax",
        clipping_bounds: tuple[float, float] | None = None,
        clip_in_scaled_space: bool = True,
        clip_to_physical_bounds: bool = False,
        max_batch: int | None = 1,
        parallel_backend: str | None = None,
        n_workers: int | None = 1,
        model_loader: Any = None,
        transition_weights_scheme: str = "linear",
        morphing_method: str = "google_film",
        apply_suppression: bool = False,
        suppression_config: dict[str, Any] | None = None,
        suppression_stages: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        """Initialise the SpatialMorphing plugin.

        Args:
            forecast_period: The forecast period in seconds used to identify the
                cluster mapping entries that define the source realizations for each
                source model.
            cluster_number: The cluster index to select and output.
            model_id_attr: Name of the cube attribute that identifies the source
                model. Defaults to "mosg__model_configuration".
            cycletime: Forecast reference time to apply to the input cubes. If set,
                the forecast periods are updated while keeping validity times fixed.
                Expected format is YYYYMMDDTHHMMZ.
            selection_attr: Optional attribute name to set on the output cube to note
                that the realization was selected by this plugin.
            selection_attr_value: Value assigned to ``selection_attr`` when it is set.
            transitions: Optional explicit transition specification. This can be
                either a dictionary containing a "transitions" list or a list of
                transition dictionaries with keys "source_a", "source_b",
                "start_forecast_period_minutes" and "end_forecast_period_minutes".
            model_path: Path to the TensorFlow Hub module used by Google FILM.
            scaling: Scaling strategy applied by Google FILM. One of "log10" or
                "minmax".
            clipping_bounds: Optional lower and upper physical bounds used when
                clipping the morphed field.
            clip_in_scaled_space: If True, clipping is applied before reverse scaling.
            clip_to_physical_bounds: If True, clipping is applied after reverse
                scaling to the physical domain.
            max_batch: Maximum batch size used for FILM inference.
            parallel_backend: Backend used for parallel processing, or None for serial
                execution.
            n_workers: Number of workers used for parallel processing.
            model_loader: Optional callable used to load the TensorFlow model.
            transition_weights_scheme: Weighting scheme used during a transition.
                Supported values are "linear" and "smoothstep".
            morphing_method: Spatial morphing backend to use for transitions.
                Supported values are "google_film" (default) and "linear".
            apply_suppression: If True, apply the local suppression workflow to the
                morphed result.
            suppression_config: Optional dictionary of tuning parameters for the
                local suppression workflow.
            suppression_stages: Optional list of suppression stages to apply. Supported
                values are "weak_signal", "convective", and "upper_tail".

        Returns:
            None.

        Raises:
            ValueError: If transition_weights_scheme is not recognised.
            ValueError: If morphing_method is not recognised.
        """
        self.forecast_period = forecast_period
        self.cluster_number = cluster_number
        self.model_id_attr = model_id_attr
        self.cycletime = cycletime
        if self.cycletime is not None:
            validate_cycletime_format(self.cycletime)
        self.selection_attr = selection_attr
        self.selection_attr_value = selection_attr_value
        self.transitions = self._parse_transitions(transitions)

        # Store Google FILM config
        self.model_path = model_path
        self.scaling = scaling
        self.clipping_bounds = _as_tuple_if_list(clipping_bounds)
        self.clip_in_scaled_space = clip_in_scaled_space
        self.clip_to_physical_bounds = clip_to_physical_bounds
        self.max_batch = max_batch
        self.parallel_backend = parallel_backend
        self.n_workers = n_workers
        self.model_loader = model_loader
        self.transition_weights_scheme = transition_weights_scheme
        self.morphing_method = morphing_method
        self.apply_suppression = apply_suppression
        self.suppression_config = suppression_config
        self.suppression_stages = suppression_stages
        # Keep suppression as a dedicated plugin so morphing orchestration and
        # suppression algorithms evolve independently.
        self._suppression_plugin = SpatialMorphingSuppression(
            suppression_config=self.suppression_config,
            suppression_stages=self.suppression_stages,
        )
        self.expected_forecast_contributors: list[dict[str, Any]] = []
        self.actual_forecast_contributors: list[dict[str, Any]] = []

        if self.transition_weights_scheme not in {"linear", "smoothstep"}:
            raise ValueError(
                "transition_weights_scheme must be 'linear' or 'smoothstep'"
            )
        if self.morphing_method not in {"google_film", "linear"}:
            raise ValueError("morphing_method must be 'google_film' or 'linear'")

        # Create RealizationSelection helper for accessing cluster mapping methods
        self._selection_helper = RealizationSelection(
            forecast_period=forecast_period,
            model_id_attr=model_id_attr,
            cycletime=cycletime,
            selection_attr=selection_attr,
            selection_attr_value=selection_attr_value,
        )

    def _parse_transitions(
        self, transitions: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        """Validate and format explicit transition definitions.

        Args:
            transitions: Transition specification to parse. This may be a dictionary
                containing a top-level "transitions" list, a list of transition
                dictionaries, or None.

        Returns:
            A formatted list of transition dictionaries with forecast-period bounds
            converted to seconds. If transitions is None, returns an empty list.

        Raises:
            TypeError: If transitions is not a dictionary, list, or None.
            TypeError: If an individual transition entry is not a dictionary.
            ValueError: If the transition data are malformed or missing required
                keys.
        """
        if transitions is None:
            return []

        if isinstance(transitions, str):
            transitions = json.loads(transitions)

        if isinstance(transitions, dict):
            if "transitions" not in transitions:
                raise ValueError(
                    "transitions dictionary must contain a 'transitions' list"
                )
            transition_list = transitions["transitions"]
        elif isinstance(transitions, list):
            transition_list = transitions
        else:
            raise TypeError(
                "transitions must be a dictionary containing a 'transitions' list "
                "or a list of transition dictionaries"
            )

        parsed_transitions: list[dict[str, Any]] = []
        for transition in transition_list:
            if not isinstance(transition, dict):
                raise TypeError("Each transition must be a dictionary")

            required_keys = {
                "source_a",
                "source_b",
                "start_forecast_period_minutes",
                "end_forecast_period_minutes",
            }
            missing = required_keys - set(transition)
            if missing:
                raise ValueError(
                    "Transition definition missing required keys: "
                    + ", ".join(sorted(missing))
                )

            source_a = str(transition["source_a"]).strip()
            source_b = str(transition["source_b"]).strip()
            if not source_a or not source_b:
                raise ValueError("Transition source names must be non-empty strings")

            start_minutes = transition["start_forecast_period_minutes"]
            end_minutes = transition["end_forecast_period_minutes"]
            if (
                not isinstance(start_minutes, int)
                or not isinstance(end_minutes, int)
                or start_minutes < 0
                or end_minutes < 0
                or start_minutes >= end_minutes
            ):
                raise ValueError(
                    "Transition start/end forecast periods must be positive integers "
                    "with start < end"
                )

            parsed_transitions.append(
                {
                    "source_a": source_a,
                    "source_b": source_b,
                    "start_forecast_period_seconds": start_minutes * 60,
                    "end_forecast_period_seconds": end_minutes * 60,
                }
            )

        return parsed_transitions

    def _find_active_transition_for_source(
        self,
        source_tag: str,
        selected_source_name: str,
        forecast_period: int,
        active_transitions: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Return the transition that matches the selected source name.

        Args:
            source_tag: Either "source_a" or "source_b" to indicate which source
                to match against.
            selected_source_name: The name of the source to match.
            forecast_period: The forecast period (in seconds) to check for active
                transitions.
            active_transitions: List of transitions that are active at the given
                forecast period.

        Returns:
            The matching transition dictionary if found, otherwise None.

        Raises:
            ValueError: If multiple transitions match the selected source name.
        """
        source_matches = [
            transition
            for transition in active_transitions
            if transition[source_tag] == selected_source_name
        ]
        if len(source_matches) == 1:
            return source_matches[0]
        if len(source_matches) > 1:
            raise ValueError(
                "Multiple transitions match forecast_period="
                f"{forecast_period} and {source_tag}={selected_source_name!r}"
            )

    @staticmethod
    def _match_transition_against_sources(
        active_transitions: list[dict[str, Any]],
        selected_source_name: str,
        available_source_names: set[str] | None,
    ) -> dict[str, Any] | None:
        """Return the first transition that is both active and compatible.

        In practice, "active" means the transition definition covers the target
        forecast period, i.e. the period lies between the transition's configured
        start and end bounds. "Compatible" means the transition can be used with the
        current source set: one side of the transition must match the selected
        source name, and the other side must be available on the input forecast
        cubes (when a source list is supplied).

        This method therefore filters the list of active transitions to those that
        are relevant to the chosen source and that do not reference a model source
        which is absent from the available forecast inputs.

        Args:
            active_transitions: Transition definitions that are active at the target
                forecast period.
            selected_source_name: Source name used to choose the correct transition
                when multiple definitions overlap.
            available_source_names: Optional set of source names present on the input
                forecast cubes; used to restrict matches to sources that are actually
                available.

        Returns:
            The matching transition dictionary, or None if no transition is both
            active and compatible with the currently available source set.
        """
        for tag_a, tag_b in (("source_b", "source_a"), ("source_a", "source_b")):
            matching = [
                transition
                for transition in active_transitions
                if transition[tag_a] == selected_source_name
                and (
                    available_source_names is None
                    or transition[tag_b] in available_source_names
                )
            ]
            if matching:
                return matching[0]

    @staticmethod
    def _format_transition_mismatch_error(
        forecast_period: int,
        selected_source_name: str,
        active_transitions: list[dict[str, Any]],
        available_source_names: set[str] | None,
    ) -> str:
        """Format an explanatory error for a missing transition match.

        Args:
            forecast_period: The forecast period (in seconds) that was checked.
            selected_source_name: The source name that was used to select a transition.
            active_transitions: List of transitions that were active at the forecast
                period.
            available_source_names: Optional set of source names present on the input
                forecast cubes.

        Returns:
            A formatted error message explaining the mismatch.
        """
        expected_transition = None
        for transition in active_transitions:
            if selected_source_name in {
                transition["source_a"],
                transition["source_b"],
            }:
                expected_transition = (
                    f"from {transition['source_a']!r} to {transition['source_b']!r}"
                )
                break

        available_sources = "{}"
        if available_source_names is not None:
            available_sources = (
                "{"
                + ", ".join(repr(name) for name in sorted(available_source_names))
                + "}"
            )

        if expected_transition is None:
            return (
                "No transition matches forecast_period="
                f"{forecast_period} for selected source {selected_source_name!r}"
            )

        return (
            "No transition matches forecast_period="
            f"{forecast_period} for selected source {selected_source_name!r}; "
            f"expected transition {expected_transition} "
            f"(available sources: {available_sources})"
        )

    def _find_active_transition(
        self,
        forecast_period: int,
        selected_source_name: str | None = None,
        available_source_names: set[str] | None = None,
    ) -> dict[str, Any] | None:
        """Return the active transition for the supplied forecast period and source.

        If multiple transitions are active at the same forecast period, the selected
        source name is used to choose between them. If available source names are
        supplied, the active transition is further constrained to those whose
        origin source is present on the input forecast cubes.

        Args:
            forecast_period: The forecast period (in seconds) to check for active
                transitions.
            selected_source_name: The destination source name to match, if multiple
                transitions are active.
            available_source_names: Source labels present on the forecast cubes.

        Returns:
            The active transition dictionary if found, otherwise None.

        Raises:
            ValueError: If multiple transitions match the forecast period and
                selected_source_name is not provided, or if no matching transition
                is found.
        """
        active_transitions = [
            transition
            for transition in self.transitions
            if transition["start_forecast_period_seconds"]
            <= forecast_period
            <= transition["end_forecast_period_seconds"]
        ]
        if len(active_transitions) <= 1:
            return active_transitions[0] if active_transitions else None

        if selected_source_name is None:
            raise ValueError(
                "Multiple transitions match forecast_period="
                f"{forecast_period}; selected_source_name is required to choose "
                "between overlapping transition definitions"
            )

        transition = self._match_transition_against_sources(
            active_transitions,
            selected_source_name,
            available_source_names,
        )
        if transition is not None:
            return transition

        if (
            available_source_names is not None
            and selected_source_name in available_source_names
        ):
            return None

        raise ValueError(
            self._format_transition_mismatch_error(
                forecast_period,
                selected_source_name,
                active_transitions,
                available_source_names,
            )
        )

    @staticmethod
    def _calculate_transition_weight(
        forecast_period: int,
        start_forecast_period_seconds: int,
        end_forecast_period_seconds: int,
    ) -> float:
        """Calculate the interpolation weight between explicit transition bounds.

        Args:
            forecast_period: The forecast period (in seconds) for which to calculate
                the weight.
            start_forecast_period_seconds: The start of the transition period (in
                seconds).
            end_forecast_period_seconds: The end of the transition period (in
                seconds).

        Returns:
            The interpolation weight as a float between 0.0 and 1.0.
        """
        if forecast_period <= start_forecast_period_seconds:
            return 0.0
        if forecast_period >= end_forecast_period_seconds:
            return 1.0

        weight = (forecast_period - start_forecast_period_seconds) / (
            end_forecast_period_seconds - start_forecast_period_seconds
        )
        return float(np.clip(weight, 0.0, 1.0))

    def _call_google_film_for_morphing(
        self,
        cube_a: Cube,
        cube_b: Cube,
        weight: float,
    ) -> Cube:
        """Use Google FILM to spatially morph between two source cubes.

        Args:
            cube_a: First source cube (weight=0).
            cube_b: Second source cube (weight=1).
            weight: Morphing weight (0.0 to 1.0).

        Returns:
            Morphed cube at the specified weight.

        Raises:
            ValueError: If weight is outside [0, 1] or if FILM config is missing.
            RuntimeError: If FILM returns no results.
        """
        if not (0.0 <= weight <= 1.0):
            raise ValueError(f"Weight must be in [0, 1], got {weight}")

        if self.model_path is None:
            raise ValueError("model_path must be provided to use Google FILM morphing")

        # Create interpolator
        interpolator = GoogleFilmInterpolation(
            model_path=self.model_path,
            scaling=self.scaling,
            clipping_bounds=self.clipping_bounds,
            clip_in_scaled_space=self.clip_in_scaled_space,
            clip_to_physical_bounds=self.clip_to_physical_bounds,
            max_batch=self.max_batch,
            parallel_backend=self.parallel_backend,
            n_workers=self.n_workers,
            model_loader=self.model_loader,
            interpolation_fractions=weight,
        )

        # Create template cube for interpolation result
        template = cube_a.copy()

        # Call FILM with weight as time_fraction. The backend returns a list of
        # interpolated cubes for the requested interpolation fractions, so we keep
        # the single result corresponding to this weight rather than returning the
        # whole collection.
        result_cubes = interpolator.process(cube_a, cube_b, template)

        if len(result_cubes) == 0:
            raise RuntimeError("Google FILM interpolation returned no results")

        return result_cubes[0]

    def _apply_morphing_backend(
        self, cube_a: Cube, cube_b: Cube, weight: float
    ) -> Cube:
        """Apply the configured morphing backend between two source cubes.

        Args:
            cube_a: First source cube.
            cube_b: Second source cube.
            weight: Morphing weight (0.0 to 1.0).

        Returns:
            Morphed cube at the specified weight.
        """
        if self.morphing_method == "google_film":
            return self._call_google_film_for_morphing(cube_a, cube_b, weight)
        if self.morphing_method == "linear":
            result = cube_a.copy()
            result.data = ((1.0 - weight) * cube_a.data + weight * cube_b.data).astype(
                np.float32
            )
            return result

    def _select_single_source_cube(
        self,
        source_name: str,
        realization_index: int | None,
        forecast_cubes: CubeList,
    ) -> Cube | None:
        """Select the single source cube needed for morphing.

        This is a convenience wrapper around the more general
        ``RealizationSelection.select_realizations_for_clusters`` helper. The
        generic helper is designed to select a list of cubes for a cluster-to-
        source mapping, whereas the morphing path needs exactly one field for one
        source and one realization at a time. This method therefore builds a
        one-entry cluster selection map, extracts the matching cube, and returns
        the result directly.

        A ``None`` return indicates that the requested source/realization pair is
        not available in the current input cube set. This is handled gracefully in
        the transition logic as a non-fatal fallback so morphing can skip an
        unavailable source rather than aborting the whole selection.

        Args:
            source_name: Source model identifier from model_id_attr.
            realization_index: Realization index to extract.
            forecast_cubes: Input forecast cubes.

        Returns:
            The selected source cube for the requested source and realization, or
            ``None`` if the source/realization cannot be extracted from the
            available inputs.
        """
        if realization_index is None:
            return None

        try:
            selected = self._selection_helper.select_realizations_for_clusters(
                {self.cluster_number: (source_name, int(realization_index))},
                forecast_cubes,
            )
        except (AttributeError, TypeError, ValueError):
            return None

        if not selected:
            return None
        return selected[0]

    @staticmethod
    def _as_contributor(
        source: str,
        realization: int | None,
        weight: float,
    ) -> dict[str, Any]:
        """Create a provenance contributor record for one blended source.

        A contributor record describes one input field that contributed to the
        final morphed output for a given transition. It is a lightweight summary of
        the source model name, the realization index selected from that source, and
        the weight assigned to that source in the blended result. These records are
        then attached to the output cube as provenance metadata so the downstream
        forecast can be traced back to the specific model/realization combination
        that contributed to each part of the transition.

        Args:
            source: Source model identifier.
            realization: Realization index, or None if not applicable.
            weight: Relative contribution of this source to the blended output.

        Returns:
            A dictionary representing one contributor entry, for example:
            {"source": "model_a", "realization": 5, "weight": 0.75}.
        """
        return {
            "source": source,
            "realization": None if realization is None else int(realization),
            "weight": float(weight),
        }

    def _diagnose_realization_for_source(
        self,
        source_name: str,
        cluster_number: int,
        target_period: int,
        secondary_map: dict[str, dict[str, list[dict[str, list[int]]]]] | None,
        primary_map: dict[str, int],
        cluster_cube: Cube,
        full_cluster_to_selection: dict[int, tuple[str, int]],
    ) -> int | None:
        """Diagnose the realization index for a source at or near a target period.

        Args:
            source_name: Source label to diagnose for.
            cluster_number: Cluster being processed.
            target_period: Forecast period in seconds used to find the most relevant
                source realization.
            secondary_map: Optional secondary realization map e.g.
                {'source_a': {'0': [{'realization': 17, 'forecast_periods':
                [475200, 518400, 561600, 604800, 648000]}]}} where the key is the
                source name, the value is a dictionary keyed by cluster number.
            primary_map: Primary cluster-to-realization mapping e.g.
                {'0': 49, '1': 33, '2': 44, '3': 29} where the key is the cluster
                number and the value is the realization index.
            cluster_cube: Cube containing the cluster mapping metadata.
            full_cluster_to_selection: Full mapping from cluster number to the
                selected source/realization pairing.

        Returns:
            The matching realization index if one can be diagnosed, otherwise None.
        """
        cluster_key = str(cluster_number)

        if secondary_map and source_name in secondary_map:
            cluster_entries = secondary_map[source_name].get(cluster_key, [])

            # Prefer exact target-period matches.
            for entry in cluster_entries:
                forecast_periods = [int(period) for period in entry["forecast_periods"]]
                if target_period in forecast_periods:
                    return int(entry["realization"])

            # Otherwise choose the closest mapped period for this source/cluster.
            nearest_realization = None
            nearest_distance = None
            for entry in cluster_entries:
                forecast_periods = [int(period) for period in entry["forecast_periods"]]
                if not forecast_periods:
                    continue
                distance = min(
                    abs(target_period - period) for period in forecast_periods
                )
                if nearest_distance is None or distance < nearest_distance:
                    nearest_distance = distance
                    nearest_realization = int(entry["realization"])
            if nearest_realization is not None:
                return nearest_realization

        # Fallback to the normal nearest-fp cluster selection if it uses this source.
        if cluster_number in full_cluster_to_selection:
            mapped_source, mapped_realization = full_cluster_to_selection[
                cluster_number
            ]
            if mapped_source == source_name:
                return int(mapped_realization)

        # Fallback to primary-map realization if this is the inferred primary model.
        primary_source = (
            self._selection_helper._extract_primary_model_from_cluster_sources(
                cluster_cube
            )
        )
        if source_name == primary_source and cluster_key in primary_map:
            return int(primary_map[cluster_key])

        return None

    def _prepare_inputs(self, *cubes: Any) -> tuple[CubeList, Cube]:
        """Flatten and validate the input cubes before morphing.

        Args:
            *cubes: Inputs passed to process; may be a single CubeList or multiple
                Cube objects.

        Returns:
            Tuple of:
            - validated forecast cubes
            - the cluster cube.
        """
        if len(cubes) == 1 and isinstance(cubes[0], CubeList):
            cubes = tuple(cubes[0])

        forecast_cubes, cluster_cube = (
            self._selection_helper.split_cubes_forecast_and_cluster(cubes)
        )
        self._selection_helper.validate_common_validity_time(forecast_cubes)

        if self.cycletime is not None:
            for cube in forecast_cubes:
                reset_forecast_reference_time_and_period(cube, self.cycletime)

        return forecast_cubes, cluster_cube

    def _resolve_cluster_selection(
        self, cluster_cube: Cube
    ) -> tuple[
        dict[int, tuple[str, int]],
        dict[str, int],
        dict[str, dict[str, list[dict[str, list[int]]]]],
    ]:
        """Resolve the cluster-to-source mapping for the selected forecast period.

        Args:
            cluster_cube: Cube containing cluster mapping metadata.

        Returns:
            A tuple of:
            - full_cluster_to_selection: mapping from cluster number to selected
              source/realization
            - primary_map: Primary cluster-to-realization mapping e.g.
                {'0': 49, '1': 33, '2': 44, '3': 29} where the key is the cluster
                number and the value is the realization index.
            - secondary_map: Secondary realization map e.g.
                {'source_a': {'0': [{'realization': 17, 'forecast_periods':
                [475200, 518400, 561600, 604800, 648000]}]}} where the key is the
                source name, the value is a dictionary keyed by cluster number.

        Raises:
            ValueError: If the requested cluster number is not found in the mapping.
        """
        primary_map, secondary_map = self._selection_helper.parse_mapping_attributes(
            cluster_cube
        )

        mapping_fps = set()
        if secondary_map:
            for cluster_dict in secondary_map.values():
                for cluster_list in cluster_dict.values():
                    for entry in cluster_list:
                        mapping_fps.update(entry["forecast_periods"])

        nearest_fp, use_secondary = (
            self._selection_helper.find_nearest_secondary_mapping_fp(
                mapping_fps, self.forecast_period
            )
        )

        full_cluster_to_selection = self._selection_helper.build_cluster_to_selection(
            nearest_fp, use_secondary, secondary_map, primary_map, cluster_cube
        )
        if self.cluster_number not in full_cluster_to_selection:
            raise ValueError(
                f"Cluster number {self.cluster_number} not found in cluster mapping."
            )

        return full_cluster_to_selection, primary_map, secondary_map

    def _resolve_cluster_selection_with_available_sources(
        self,
        cluster_to_selection: dict[int, tuple[str, int]],
        forecast_cubes: CubeList,
        cluster_cube: Cube,
        primary_map: dict[str, int],
        secondary_map: dict[str, dict[str, list[dict[str, list[int]]]]] | None,
    ) -> dict[int, tuple[str, int]]:
        """Fall back to an available source if the mapped source is absent or invalid

        Args:
            cluster_to_selection: Mapping from cluster number to selected source and
                realization.
            forecast_cubes: List of forecast cubes to select from.
            cluster_cube: Cube containing the cluster mapping metadata.
            primary_map: Primary cluster-to-realization mapping e.g.
                {'0': 49, '1': 33, '2': 44, '3': 29} where the key is the cluster
                number and the value is the realization index.
            secondary_map: Secondary realization map e.g.
                {'source_a': {'0': [{'realization': 17, 'forecast_periods':
                [475200, 518400, 561600, 604800, 648000]}]}} where the key is the
                source name, the value is a dictionary keyed by cluster number.

        Returns:
            Updated cluster_to_selection mapping with a valid source and realization
            for the requested cluster number.
        """
        available_source_names = {
            cube.attributes.get(self.model_id_attr)
            for cube in forecast_cubes
            if cube.attributes.get(self.model_id_attr) is not None
        }
        requested_source, requested_realization = cluster_to_selection[
            self.cluster_number
        ]

        requested_cube = forecast_cubes.extract(
            iris.AttributeConstraint(**{self.model_id_attr: requested_source})
        )
        has_valid_requested_realization = False
        if requested_cube:
            requested_model_cube = requested_cube[0]
            if not requested_model_cube.coords("realization"):
                has_valid_requested_realization = True
            elif (
                requested_realization
                in requested_model_cube.coord("realization").points
            ):
                has_valid_requested_realization = True

        if (
            requested_source in available_source_names
            and has_valid_requested_realization
        ):
            return cluster_to_selection

        return self._choose_fallback_cluster_source(
            cluster_to_selection,
            forecast_cubes,
            cluster_cube,
            primary_map,
            secondary_map,
            requested_source,
            requested_realization,
            available_source_names,
            has_valid_requested_realization,
        )

    def _choose_fallback_cluster_source(
        self,
        cluster_to_selection: dict[int, tuple[str, int]],
        forecast_cubes: CubeList,
        cluster_cube: Cube,
        primary_map: dict[str, int],
        secondary_map: dict[str, dict[str, list[dict[str, list[int]]]]] | None,
        requested_source: str,
        requested_realization: int,
        available_source_names: set[str],
        has_valid_requested_realization: bool,
    ) -> dict[int, tuple[str, int]]:
        """Select the first valid fallback source and realization for the cluster.

        Fallback realization diagnosis is source-aware: it first uses the
        secondary map for the candidate fallback source, and only if that is
        unavailable does it fall back to the primary-map realization for the
        cluster.

        Args:
            cluster_to_selection: Mapping from cluster number to selected source and
                realization.
            forecast_cubes: List of forecast cubes to select from.
            cluster_cube: Cube containing the cluster mapping metadata.
            primary_map: Primary cluster-to-realization mapping e.g.
                {'0': 49, '1': 33, '2': 44, '3': 29} where the key is the cluster
                number and the value is the realization index.
            secondary_map: Secondary realization map e.g.
                {'source_a': {'0': [{'realization': 17, 'forecast_periods':
                [475200, 518400, 561600, 604800, 648000]}]}} where the key is the
                source name, the value is a dictionary keyed by cluster number.
            requested_source: The source that was originally requested for the cluster.
            requested_realization: The realization that was originally requested for
                the cluster.
            available_source_names: Set of source names present on the forecast cubes.
            has_valid_requested_realization: Whether the requested source and
                realization are valid and present on the forecast cubes.

        Returns:
            Updated cluster_to_selection mapping with a valid source and realization
            for the requested cluster number.

        Warns:
            If the requested source and realization are unavailable and no valid
            fallback source is found, a warning is issued.
            If the requested source is unavailable but a valid fallback source is
            found, a warning is issued indicating the fallback source and
            realization being used.
        """
        cluster_key = str(self.cluster_number)
        cluster_sources = {}
        if "cluster_sources" in cluster_cube.attributes:
            cluster_sources = json.loads(cluster_cube.attributes["cluster_sources"])

        candidate_sources = [
            source_name
            for source_name in cluster_sources.get(cluster_key, {})
            if source_name in available_source_names and source_name != requested_source
        ]
        if not candidate_sources:
            candidate_sources = sorted(
                source_name
                for source_name in available_source_names
                if source_name != requested_source
            )

        if not candidate_sources:
            warnings.warn(
                f"Cluster {self.cluster_number} requested source {requested_source!r} "
                f"realization {requested_realization} is unavailable and no fallback "
                "source is available.",
                UserWarning,
            )
            return cluster_to_selection

        primary_realization = None
        if cluster_key in primary_map:
            primary_realization = int(primary_map[cluster_key])

        for fallback_source in candidate_sources:
            fallback_cube = forecast_cubes.extract(
                iris.AttributeConstraint(**{self.model_id_attr: fallback_source})
            )
            if not fallback_cube:
                continue

            fallback_model_cube = fallback_cube[0]
            fallback_realization = None

            if secondary_map and fallback_source in secondary_map:
                fallback_realization = self._diagnose_realization_for_source(
                    source_name=fallback_source,
                    cluster_number=self.cluster_number,
                    target_period=self.forecast_period,
                    secondary_map=secondary_map,
                    primary_map=primary_map,
                    cluster_cube=cluster_cube,
                    full_cluster_to_selection=cluster_to_selection,
                )

            if fallback_realization is None:
                fallback_realization = primary_realization

            if fallback_realization is None:
                continue

            if not fallback_model_cube.coords("realization"):
                warnings.warn(
                    f"Cluster {self.cluster_number} requested source "
                    f"{requested_source!r} realization {requested_realization} is "
                    f"unavailable; using fallback source {fallback_source!r} "
                    f"realization {fallback_realization}.",
                    UserWarning,
                )
                cluster_to_selection[self.cluster_number] = (
                    fallback_source,
                    fallback_realization,
                )
                return cluster_to_selection

            realization_points = fallback_model_cube.coord("realization").points
            if int(fallback_realization) in realization_points:
                warnings.warn(
                    f"Cluster {self.cluster_number} requested source "
                    f"{requested_source!r} realization {requested_realization} is "
                    f"unavailable; using fallback source {fallback_source!r} "
                    f"realization {int(fallback_realization)}.",
                    UserWarning,
                )
                cluster_to_selection[self.cluster_number] = (
                    fallback_source,
                    int(fallback_realization),
                )
                return cluster_to_selection

        if (
            requested_source not in available_source_names
            or not has_valid_requested_realization
        ):
            warnings.warn(
                f"Cluster {self.cluster_number} requested source {requested_source!r} "
                f"realization {requested_realization} is unavailable and no valid "
                "fallback source was found.",
                UserWarning,
            )
        return cluster_to_selection

    def _select_transition_source_cubes(
        self,
        active_transition: dict[str, Any],
        forecast_cubes: CubeList,
        cluster_number: int,
        secondary_map: dict[str, dict[str, list[dict[str, list[int]]]]] | None,
        primary_map: dict[str, int],
        cluster_cube: Cube,
        full_cluster_to_selection: dict[int, tuple[str, int]],
    ) -> tuple[Cube | None, Cube | None, float | None, int | None, int | None]:
        """Select the source cubes and transition weight for the active transition.

        Args:
            active_transition: Active transition definition for the target forecast
                period.
            forecast_cubes: Available forecast cubes.
            cluster_number: Requested cluster number.
            secondary_map: Secondary realization map e.g.
                {'source_a': {'0': [{'realization': 17, 'forecast_periods':
                [475200, 518400, 561600, 604800, 648000]}]}} where the key is the
                source name, the value is a dictionary keyed by cluster number.
            primary_map: Primary cluster-to-realization mapping e.g.
                {'0': 49, '1': 33, '2': 44, '3': 29} where the key is the cluster
                number and the value is the realization index.
            cluster_cube: Cube containing cluster metadata.
            full_cluster_to_selection: Mapping from every cluster number to the
                selected source and realization for that cluster, e.g.
                {0: ("uk_ens", 8), 1: ("uk_ens", 7), 2: ("uk_ens", 6)}. This is
                the comprehensive cluster-to-source selection table used to resolve
                source/realization choices across all clusters; the current cluster
                number is then selected from this table when a transition is being
                evaluated.

        Returns:
            A tuple of ``(cube_a, cube_b, weight, source_a_realization,
            source_b_realization)``, where weight is the morphing weight for the
            active transition or None if no transition is being applied.
        """
        start_forecast_period_seconds = active_transition[
            "start_forecast_period_seconds"
        ]
        end_forecast_period_seconds = active_transition["end_forecast_period_seconds"]
        source_a = active_transition["source_a"]
        source_b = active_transition["source_b"]

        source_a_realization = self._diagnose_realization_for_source(
            source_name=source_a,
            cluster_number=cluster_number,
            target_period=start_forecast_period_seconds,
            secondary_map=secondary_map,
            primary_map=primary_map,
            cluster_cube=cluster_cube,
            full_cluster_to_selection=full_cluster_to_selection,
        )
        source_b_realization = self._diagnose_realization_for_source(
            source_name=source_b,
            cluster_number=cluster_number,
            target_period=end_forecast_period_seconds,
            secondary_map=secondary_map,
            primary_map=primary_map,
            cluster_cube=cluster_cube,
            full_cluster_to_selection=full_cluster_to_selection,
        )

        cube_a = self._select_single_source_cube(
            source_a,
            source_a_realization,
            forecast_cubes,
        )
        cube_b = self._select_single_source_cube(
            source_b,
            source_b_realization,
            forecast_cubes,
        )

        if cube_a is None or cube_b is None:
            return None, None, None, source_a_realization, source_b_realization

        weight = self._calculate_transition_weight(
            self.forecast_period,
            start_forecast_period_seconds,
            end_forecast_period_seconds,
        )
        if self.transition_weights_scheme == "smoothstep":
            weight = weight * weight * (3.0 - 2.0 * weight)

        return (
            cube_a,
            cube_b,
            float(weight),
            source_a_realization,
            source_b_realization,
        )

    def _finalise_output_cube(self, result_cube: Cube) -> Cube:
        """Apply final attribute and coordinate cleanup before returning output.

        Args:
            result_cube: Cube to finalise.

        Returns:
            Finalised cube with cleaned-up attributes and coordinates.
        """
        result_cube = remove_blend_time(result_cube)
        result_cube = remove_deprecation_warnings(result_cube)

        if result_cube.coords("realization"):
            result_cube.coord("realization").points = [self.cluster_number]
            result_cube.coord("realization").units = "1"

        result_cube.attributes.pop(self.model_id_attr, None)
        result_cube.attributes["expected_forecast_contributors"] = json.dumps(
            self.expected_forecast_contributors
        )
        result_cube.attributes["actual_forecast_contributors"] = json.dumps(
            self.actual_forecast_contributors
        )

        if self.selection_attr is not None:
            result_cube.attributes[self.selection_attr] = self.selection_attr_value

        return result_cube

    def _diagnose_expected_morphing_contributions(
        self,
        expected_selected_source: str,
        expected_selected_realization: int,
        full_cluster_to_selection: dict[int, tuple[str, int]],
        primary_map: dict[str, int],
        secondary_map: dict[str, dict[str, list[dict[str, list[int]]]]] | None,
        cluster_cube: Cube,
    ) -> None:
        """Set expected provenance metadata for the selected cluster and forecast.

        This method determines whether the selected cluster falls within an active
        transition and, if so, records the expected source/realization
        contribution(s) that should appear in the final blended output. The values
        are stored in ``self.expected_forecast_contributors`` for later provenance
        reporting.

        Args:
            expected_selected_source: Source model selected by the initial cluster
                mapping.
            expected_selected_realization: Realization selected by the initial
                cluster mapping.
            full_cluster_to_selection: Full cluster-to-source/realization lookup.
            primary_map: Primary cluster-to-realization mapping e.g.
                {'0': 49, '1': 33, '2': 44, '3': 29} where the key is the cluster
                number and the value is the realization index.
            secondary_map: Secondary realization map e.g.
                {'source_a': {'0': [{'realization': 17, 'forecast_periods':
                [475200, 518400, 561600, 604800, 648000]}]}} where the key is the
                source name, the value is a dictionary keyed by cluster number.
            cluster_cube: Cube containing the cluster metadata used to diagnose
                source/realization selections.

        Returns:
            None. This method updates expected contributor metadata on the
            plugin instance.
        """
        self.expected_forecast_contributors = [
            self._as_contributor(
                expected_selected_source,
                expected_selected_realization,
                1.0,
            )
        ]
        self.actual_forecast_contributors = []

        try:
            expected_transition = self._find_active_transition(
                self.forecast_period,
                selected_source_name=expected_selected_source,
            )
        except ValueError:
            expected_transition = None
        if expected_transition is None:
            return

        source_a = expected_transition["source_a"]
        source_b = expected_transition["source_b"]
        source_a_realization = self._diagnose_realization_for_source(
            source_name=source_a,
            cluster_number=self.cluster_number,
            target_period=expected_transition["start_forecast_period_seconds"],
            secondary_map=secondary_map,
            primary_map=primary_map,
            cluster_cube=cluster_cube,
            full_cluster_to_selection=full_cluster_to_selection,
        )
        source_b_realization = self._diagnose_realization_for_source(
            source_name=source_b,
            cluster_number=self.cluster_number,
            target_period=expected_transition["end_forecast_period_seconds"],
            secondary_map=secondary_map,
            primary_map=primary_map,
            cluster_cube=cluster_cube,
            full_cluster_to_selection=full_cluster_to_selection,
        )
        expected_weight = self._calculate_transition_weight(
            self.forecast_period,
            expected_transition["start_forecast_period_seconds"],
            expected_transition["end_forecast_period_seconds"],
        )
        if self.transition_weights_scheme == "smoothstep":
            expected_weight = (
                expected_weight * expected_weight * (3.0 - 2.0 * expected_weight)
            )

        if expected_weight <= 0.0:
            self.expected_forecast_contributors = [
                self._as_contributor(source_a, source_a_realization, 1.0)
            ]
        elif expected_weight >= 1.0:
            self.expected_forecast_contributors = [
                self._as_contributor(source_b, source_b_realization, 1.0)
            ]
        else:
            self.expected_forecast_contributors = [
                self._as_contributor(
                    source_a, source_a_realization, 1.0 - expected_weight
                ),
                self._as_contributor(source_b, source_b_realization, expected_weight),
            ]

    def _apply_morphing(
        self,
        result_cube: Cube,
        cluster_to_selection: dict[int, tuple[str, int]],
        forecast_cubes: CubeList,
        cluster_cube: Cube,
        full_cluster_to_selection: dict[int, tuple[str, int]],
        primary_map: dict[str, int],
        secondary_map: dict[str, dict[str, list[dict[str, list[int]]]]] | None,
    ) -> Cube:
        """Apply any active transition morphing and update provenance metadata.

        This method resolves the actual source selected for the current cluster,
        checks whether the forecast period lies within a configured transition,
        and, if so, applies the active source transition logic (including optional
        local suppression) before returning the final cube for provenance
        finalisation.

        Args:
            result_cube: The cube selected before any transition remapping.
            cluster_to_selection: Selected source and realization for the current
                cluster after any fallback logic.
            forecast_cubes: Available forecast cubes.
            cluster_cube: Cube containing the cluster metadata.
            full_cluster_to_selection: Full cluster-to-source/realization lookup.
            primary_map: Primary cluster-to-realization mapping e.g.
                {'0': 49, '1': 33, '2': 44, '3': 29} where the key is the cluster
                number and the value is the realization index.
            secondary_map: Secondary realization map e.g.
                {'source_a': {'0': [{'realization': 17, 'forecast_periods':
                [475200, 518400, 561600, 604800, 648000]}]}} where the key is the
                source name, the value is a dictionary keyed by cluster number.

        Returns:
            The cube after the active transition has been applied (if any), or the
            original result cube when no transition is active.
        """
        selected_source_name, selected_realization = cluster_to_selection[
            self.cluster_number
        ]
        self.actual_forecast_contributors = [
            self._as_contributor(selected_source_name, selected_realization, 1.0)
        ]

        available_source_names = {
            cube.attributes.get(self.model_id_attr)
            for cube in forecast_cubes
            if cube.attributes.get(self.model_id_attr) is not None
        }
        active_transition = self._find_active_transition(
            self.forecast_period,
            selected_source_name=selected_source_name,
            available_source_names=available_source_names,
        )

        if active_transition is None:
            return result_cube

        (
            cube_a,
            cube_b,
            weight,
            source_a_realization,
            source_b_realization,
        ) = self._select_transition_source_cubes(
            active_transition,
            forecast_cubes,
            self.cluster_number,
            secondary_map,
            primary_map,
            cluster_cube,
            full_cluster_to_selection,
        )

        if cube_a is None or cube_b is None:
            return result_cube

        if weight is not None and weight <= 0.0:
            result_cube = cube_a
            self.actual_forecast_contributors = [
                self._as_contributor(
                    active_transition["source_a"], source_a_realization, 1.0
                )
            ]
        elif weight is not None and weight >= 1.0:
            result_cube = cube_b
            self.actual_forecast_contributors = [
                self._as_contributor(
                    active_transition["source_b"], source_b_realization, 1.0
                )
            ]
        else:
            result_cube = self._apply_morphing_backend(
                cube_a,
                cube_b,
                weight,
            )
            self.actual_forecast_contributors = [
                self._as_contributor(
                    active_transition["source_a"],
                    source_a_realization,
                    1.0 - weight,
                ),
                self._as_contributor(
                    active_transition["source_b"], source_b_realization, weight
                ),
            ]

        if self.apply_suppression and weight is not None and 0.0 < weight < 1.0:
            result_cube = self._suppression_plugin.process(
                result_cube,
                cube_a,
                cube_b,
                weight=weight,
            )

        return result_cube

    def process(self, *cubes: Any) -> Cube:
        """Select realizations from forecast sources and apply spatial morphing.

        Uses RealizationSelection workflow to identify which realization from
        which forecast source corresponds to the requested cluster_number, then
        applies Google FILM spatial morphing if needed to create seamless blends.

        Args:
            *cubes: Input cubes (CubeList or multiple Cube objects) containing
                forecast cubes from different sources (all at same validity time)
                and a cluster cube with mapping attributes from
                RealizationClusterAndMatch.

        Returns:
            Single Cube containing the selected and (if applicable) spatially
            morphed realization, relabelled to cluster_number.

        Raises:
            RuntimeError: If no realization is selected for the requested cluster.
        """
        forecast_cubes, cluster_cube = self._prepare_inputs(*cubes)
        # full_cluster_to_selection is the full lookup table for every cluster,
        # mapping each cluster number to the selected source and realization used
        # for that cluster at the chosen forecast period.
        full_cluster_to_selection, primary_map, secondary_map = (
            self._resolve_cluster_selection(cluster_cube)
        )

        # cluster_to_selection restricts that full lookup to the current cluster.
        cluster_to_selection = {
            self.cluster_number: full_cluster_to_selection[self.cluster_number]
        }
        expected_selected_source, expected_selected_realization = cluster_to_selection[
            self.cluster_number
        ]

        # self.expected_forecast_contributors records the contribution(s) that we
        # expect to be present in the final blended output before the actual
        # source cubes are selected and morphed.
        self._diagnose_expected_morphing_contributions(
            expected_selected_source,
            expected_selected_realization,
            full_cluster_to_selection,
            primary_map,
            secondary_map,
            cluster_cube,
        )

        # cluster_to_selection may be updated here to use a valid fallback source
        # and realization if the initially mapped source is absent or invalid.
        cluster_to_selection = self._resolve_cluster_selection_with_available_sources(
            cluster_to_selection,
            forecast_cubes,
            cluster_cube,
            primary_map,
            secondary_map,
        )
        selected_cubes = self._selection_helper.select_realizations_for_clusters(
            cluster_to_selection, forecast_cubes
        )
        if len(selected_cubes) == 0:
            raise RuntimeError(
                f"No realization selected for cluster {self.cluster_number}"
            )

        result_cube = selected_cubes[0]
        # self.actual_forecast_contributors is updated inside _apply_morphing
        # to reflect the source(s) that actually contributed to the final output
        # after any active transition and fallback resolution.
        result_cube = self._apply_morphing(
            result_cube,
            cluster_to_selection,
            forecast_cubes,
            cluster_cube,
            full_cluster_to_selection,
            primary_map,
            secondary_map,
        )

        return self._finalise_output_cube(result_cube)


class SpatialMorphingSuppression(BasePlugin):
    """Apply local precipitation suppression stages to a morphed transition field.

    This plugin encapsulates the suppression workflow used after spatial morphing,
    including weak-signal damping, convective restoration, and upper-tail
    restoration. It is designed specifically for precipitation-like fields where
    wet-threshold and intensity-tail logic is meaningful. The implementation does
    not enforce a diagnostic-name check because multiple precipitation diagnostics
    (for example, accumulation or rate) may be valid inputs. Callers are therefore
    expected to pass precipitation diagnostics only.
    """

    def __init__(
        self,
        suppression_config: dict[str, Any] | None = None,
        suppression_stages: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        """Initialise suppression settings for morphed fields.

        Args:
            suppression_config: Optional suppression tuning values.
            suppression_stages: Optional suppression stage list.

        Returns:
            None.
        """
        self.suppression_config = self.validate_suppression_config(suppression_config)
        self.suppression_stages = self.validate_suppression_stages(suppression_stages)

    @classmethod
    def validate_suppression_config(
        cls,
        suppression_config: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Validate suppression tuning values from config data or defaults.

        Args:
            suppression_config: Optional dictionary of suppression settings. If
                None, the class defaults are used.

        Returns:
            Dictionary containing a complete suppression configuration with all
            expected keys populated.

        Raises:
            TypeError: If suppression_config is not a dictionary or None.
            ValueError: If suppression_config contains unsupported keys.
        """
        if suppression_config is None:
            return _SUPPRESSION_DEFAULTS.copy()
        if not isinstance(suppression_config, dict):
            raise TypeError(
                "suppression_config must be a dictionary or None, "
                f"got {type(suppression_config).__name__}"
            )

        merged = _SUPPRESSION_DEFAULTS.copy()
        unknown = sorted(set(suppression_config) - set(_SUPPRESSION_DEFAULTS))
        if unknown:
            raise ValueError(
                "Unknown suppression_config entries: " + ", ".join(unknown)
            )

        merged.update(suppression_config)
        return merged

    @classmethod
    def validate_suppression_stages(
        cls,
        suppression_stages: list[str] | tuple[str, ...] | None,
    ) -> tuple[str, ...]:
        """Validate the requested suppression stages and return canonical names.

        If no suppression stages are specified, an empty tuple is returned so that
        no suppression is applied by default.

        Args:
            suppression_stages: Optional stage names requested by the caller.

        Returns:
            Tuple of canonical stage names in execution order. An empty tuple means
            that suppression is disabled.

        Raises:
            ValueError: If an unsupported stage name is provided.
        """
        if suppression_stages is None:
            return ()

        if isinstance(suppression_stages, str):
            suppression_stages = [suppression_stages]

        cleaned = {
            str(stage).strip()
            for stage in suppression_stages
            if stage is not None and str(stage).strip()
        }

        if not cleaned:
            return ()

        if "all" in cleaned:
            return _SUPPRESSION_CANONICAL_STAGES

        unsupported = sorted(cleaned - set(_SUPPRESSION_CANONICAL_STAGES))
        if unsupported:
            raise ValueError(
                "Unsupported suppression stage: "
                f"{unsupported[0]}. Supported values are "
                f"{', '.join(_SUPPRESSION_CANONICAL_STAGES)}."
            )

        return tuple(
            stage for stage in _SUPPRESSION_CANONICAL_STAGES if stage in cleaned
        )

    @staticmethod
    def _validate_suppression_settings(settings: dict[str, Any]) -> None:
        """Validate merged suppression settings and transition weight.

        Args:
            settings: Fully populated suppression settings dictionary.

        Returns:
            None.

        Raises:
            ValueError: If any validated setting is outside its accepted range.
        """

        def check_in_range(name: str, value: float, lower: float, upper: float) -> None:
            if not lower <= value <= upper:
                lower_label = "0" if lower == 0.0 and upper == 1.0 else str(lower)
                upper_label = "1" if lower == 0.0 and upper == 1.0 else str(upper)
                raise ValueError(
                    f"{name} must lie in [{lower_label}, {upper_label}], got {value}"
                )

        def check_positive(name: str, value: float) -> None:
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")

        def check_minimum(name: str, value: float, minimum: float) -> None:
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}, got {value}")

        check_in_range(
            "occurrence_threshold",
            settings["occurrence_threshold"],
            0.0,
            np.inf,
        )

        for key in (
            "quantile_for_centre",
            "maximum_suppression",
            "showery_weight_factor",
            "weakness_weight_factor",
            "convective_gain",
            "upper_tail_quantile",
            "convective_mask_threshold",
            "upper_tail_intensity_quantile",
        ):
            check_in_range(key, settings[key], 0.0, 1.0)

        for key in (
            "width_fraction",
            "showery_neighbourhood_size",
            "convective_neighbourhood_size",
            "concentration_scale",
            "intensity_weight_width_fraction",
            "sigmoid_clip_limit",
        ):
            check_positive(key, settings[key])

        check_minimum(
            "maximum_intensity_scale",
            settings["maximum_intensity_scale"],
            1.0,
        )

    @staticmethod
    def _sanitize_valid_data(data: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """Replace invalid values with zeros while leaving valid points unchanged."""
        finite_data = np.nan_to_num(
            np.asarray(data, dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return np.where(valid_mask, finite_data, 0.0)

    def _compute_weak_signal_suppression(
        self,
        result_data: np.ndarray,
        source_a_data: np.ndarray,
        source_b_data: np.ndarray,
        weighted_reference: np.ndarray,
        valid_mask: np.ndarray,
        weight: float,
        threshold: float,
        quantile_for_centre: float,
        width_fraction: float,
        maximum_suppression: float,
        showery_neighbourhood_size: int,
        showery_weight_factor: float,
        weakness_weight_factor: float,
        sigmoid_clip_limit: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reduce broad weak wet halos while retaining coherent source-supported rain.

        This stage damps the portion of the morphed field that sits above a source-
        weighted reference. The reference is a weighted average of the two source
        fields, so the suppression acts only on excess precipitation that falls
        outside the local source-supported envelope rather than on the whole field.

        The motivation is to avoid Google FILM turning sparse showers into a broad
        area of light precipitation. In other words, the method suppresses the weak,
        diffuse excess that can appear around the transition without removing the
        coherent rain that is still supported by the source fields.

        The strength of the correction is based on two complementary indicators:

        1. A weak-signal diagnostic that compares the morphed intensity with a
           source-derived central quantile. This identifies the broad, diffuse areas
           where the transition is only weakly supported by the source data.
        2. A local showery diagnostic based on wet-occurrence in the neighbouring
           source pixels. This increases the suppression where the signal looks
           spatially diffuse or low confidence, rather than strongly tied to the
           source precipitation structure.

        The term "logistic" here refers to a smooth sigmoid-shaped transfer
        function, not a fitted logistic-regression model. It is used because the
        suppression should change gradually as the signal moves away from the
        source-supported centre, rather than switching abruptly at a hard
        threshold. As the morphed intensity rises above that centre, the function
        transitions smoothly from weak support toward stronger confidence, which is
        then converted into a capped suppression amount.

        The method does not force the field back below the weighted reference in a
        hard step. Instead, it reduces only the positive excess by a capped amount,
        which helps remove weak, widespread FILM artefacts while preserving
        coherent rain structures that are supported by the source fields.

        Example:
            Suppose ``weight=0.5``, source A is 100, source B is 200, and the
            morphed value is 170 at one point. The weighted reference is
            ``0.5 * 100 + 0.5 * 200 = 150``, so the excess is ``170 - 150 = 20``.
            If the combined correction fraction at that point is 0.4, the output is
            ``170 - 0.4 * 20 = 162``. If the correction fraction were 0, the point
            would remain 170.

        Args:
            result_data: Morphed output data before suppression.
            source_a_data: Source-a data array.
            source_b_data: Source-b data array.
            weighted_reference: Weighted source reference data already calculated in
                the parent process method.
            valid_mask: Boolean mask for points valid across all inputs.
            weight: Morphing weight in the range [0, 1].
            threshold: Wet-occurrence threshold for the source fields.
            quantile_for_centre: Quantile of the valid source signal used to set the
                centre of the smooth sigmoid transition.
            width_fraction: Fractional width of the sigmoid around that centre.
            maximum_suppression: Upper bound on the suppression fraction applied to
                excess precipitation.
            showery_neighbourhood_size: Neighbourhood size for the showery diagnosis.
            showery_weight_factor: Weight applied to the showery term in the final
                correction.
            weakness_weight_factor: Weight applied to the weak-signal term in the
                final correction.
            sigmoid_clip_limit: Maximum absolute value used to clip the normalised
                sigmoid input before exponentiation.

        Returns:
            Tuple of:
            - suppression-adjusted output data array
            - weighted source reference array.
        """

        # Identify wet points in each source field, including their local context.
        occ_a = source_a_data > threshold
        occ_b = source_b_data > threshold
        wet_a_mask = valid_mask & (source_a_data > threshold)
        wet_b_mask = valid_mask & (source_b_data > threshold)
        source_signal = np.concatenate(
            (source_a_data[wet_a_mask], source_b_data[wet_b_mask])
        )

        output_data = result_data.copy()
        if source_signal.size == 0:
            # If neither source field contains any valid values above the wet
            # threshold, then source_signal is empty and the code takes this
            # fallback branch. We zero only the valid data points because there is
            # no source-supported wet signal to suppress.
            output_data[valid_mask] = 0.0
            return output_data, weighted_reference

        # Set the smooth transition centre from the wet source signal, ensuring it
        # stays above the wet threshold.
        centre = float(np.quantile(source_signal, quantile_for_centre, method="linear"))
        centre = max(
            centre,
            np.nextafter(
                np.float64(threshold),
                np.float64(np.inf),
            ),
        )
        width = max(width_fraction * centre, 10.0 * np.finfo(np.float64).eps)

        # The sigmoid maps the morphed intensity to a value in [0, 1]: low values
        # indicate weak, diffuse precipitation that should be suppressed most.
        # The sigmoid is clipped to avoid numerical overflow in the exponentiation.
        logistic_argument = np.clip(
            (result_data - centre) / width, -sigmoid_clip_limit, sigmoid_clip_limit
        )
        film_fraction = 1.0 / (1.0 + np.exp(-logistic_argument))
        weakness = 1.0 - film_fraction

        # This checks whether the surrounding area is mostly dry. The uniform filter
        # computes a neighbourhood mean of the wet-occurrence masks, which is then
        # inverted to give a showery weight in [0, 1] where larger values indicate
        # more diffuse, showery precipitation that should be suppressed more strongly.
        local_occ_a = uniform_filter(
            occ_a.astype(np.float32), size=showery_neighbourhood_size
        )
        local_occ_b = uniform_filter(
            occ_b.astype(np.float32), size=showery_neighbourhood_size
        )
        showery_a = 1.0 - local_occ_a
        showery_b = 1.0 - local_occ_b
        showery_weight = (1.0 - weight) * showery_a + weight * showery_b

        # Combine the weak-signal and showery diagnostics into a capped
        # correction fraction.
        correction_fraction = (weakness_weight_factor * weakness) + (
            showery_weight_factor * showery_weight
        )
        correction_fraction = np.clip(correction_fraction, 0.0, maximum_suppression)

        # The excess represents how much the Google FILM interpolated field exceeds
        # the weighted average reference. The excess is subtracted from the Google FILM
        # interpolated field but using a correction fraction to control which
        # precipitation should be suppressed. The correction fraction is comprised of
        # a combination of two terms: a weak signal term, which is targeted at broad
        # weak precipitation, and a showery term, which is also targeted at diffuse
        # precipitation that is not well supported by the surrounding area.
        excess = np.maximum(result_data - weighted_reference, 0.0)
        output_data = result_data - correction_fraction * excess
        return output_data, weighted_reference

    @staticmethod
    def _diagnose_convective_weight(
        result_data: np.ndarray,
        source_a_data: np.ndarray,
        source_b_data: np.ndarray,
        weighted_reference: np.ndarray,
        valid_mask: np.ndarray,
        threshold: float,
        convective_neighbourhood_size: int,
        convective_gain: float,
        concentration_reference: float,
        concentration_scale: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Diagnose where the source fields are locally convective or shower-like.

        This stage looks for small, intense precipitation structures that are still
        present in the source fields but may have been weakened by the earlier broad
        weak-signal suppression. It compares the local maximum in a neighbourhood to
        the local mean, which gives a simple measure of how concentrated the rain is.

        A larger value means the precipitation is more sharply peaked, which usually
        indicates a convective or shower-like structure. The diagnosed weight is then
        used to nudge the output back toward the higher of the current result and the
        weighted source reference, but only in those locally concentrated areas. If
        the current FILM result is already stronger than the source-informed target,
        no additional uplift is applied and the field is left unchanged.

        Example:
            Suppose at one point ``result_data=30`` and ``weighted_reference=50`` so
            the local target is 50. If local concentration gives
            ``convective_weight=0.8`` and ``convective_gain=0.5``, the uplift is
            ``0.5 * 0.8 * (50 - 30) = 8`` and the corrected value becomes 38.
            If instead ``result_data=55``, the target is 55 and
            ``target - result_data = 0``, so no convective uplift is applied.

        Args:
            result_data: Current output data to adjust.
            source_a_data: Source-a data array.
            source_b_data: Source-b data array.
            weighted_reference: Weighted source reference data.
            valid_mask: Boolean mask for points valid across all inputs.
            threshold: Minimum allowed denominator when comparing local max and mean.
            convective_neighbourhood_size: Size of the neighbourhood used for the local
                statistics.
            convective_gain: Strength of the local restoration in convective areas.
            concentration_reference: Baseline concentration value for the start of the
                convective response.
            concentration_scale: Spread of the concentration-to-weight conversion.

        Returns:
            Tuple of:
            - output data with local convective intensity restored
            - diagnosed convective weight array.
        """
        # Mask invalid points before the local neighbourhood statistics so NaNs and
        # infs do not contaminate the local max/mean calculations.
        source_peak = np.maximum(source_a_data, source_b_data)
        source_peak = np.where(valid_mask, source_peak, 0.0)
        local_source_max = maximum_filter(
            source_peak, size=convective_neighbourhood_size
        )
        local_source_mean = uniform_filter(
            source_peak, size=convective_neighbourhood_size
        )

        # A ratio greater than 1 indicates a sharp, local peak rather than broad weak
        # precipitation. The weight is then scaled to [0, 1] so it can be used as a
        # local restoration factor.
        concentration = local_source_max / np.maximum(local_source_mean, threshold)
        convective_weight = np.clip(
            (concentration - concentration_reference) / concentration_scale, 0.0, 1.0
        )
        convective_weight = np.where(valid_mask, convective_weight, 0.0)

        # Restore intensity only where the source fields still show a concentrated
        # convective signal. If the morphing result is already stronger than the
        # source-informed target, then target - result_data is zero and no extra
        # uplift is applied. This preserves locally strong FILM peaks while only
        # boosting pixels where the source data suggest a genuine shower signal is
        # missing from the current morph.
        target = np.maximum(result_data, weighted_reference)
        corrected_data = result_data.copy()
        corrected_data[valid_mask] = result_data[valid_mask] + convective_gain * (
            convective_weight[valid_mask]
            * (target[valid_mask] - result_data[valid_mask])
        )
        return corrected_data, convective_weight

    @staticmethod
    def _restore_upper_tail(
        result_data: np.ndarray,
        source_a_data: np.ndarray,
        source_b_data: np.ndarray,
        valid_mask: np.ndarray,
        convective_weight: np.ndarray,
        weight: float,
        threshold: float,
        upper_tail_quantile: float,
        convective_mask_threshold: float,
        maximum_intensity_scale: float,
        upper_tail_intensity_quantile: float,
        intensity_weight_width_fraction: float,
        sigmoid_clip_limit: float,
    ) -> np.ndarray:
        """Restore intense precipitation where the convective signal is still weak.

        This stage compares a source-derived high-end value with the current
        morphed field in locally convective, wet areas. The source-derived value is
        calculated from the wet occurrences in both source fields at a chosen
        upper-tail quantile, and the current value is calculated from the same
        quantile in the corresponding Google FILM output.

        If the current field is weaker than the expected source signal, the ratio
        between the two values gives a local scale factor. That scale is then
        applied only to the strongest wet pixels, with a cap on the maximum allowed
        increase so the correction stays local and physically realistic.

        Example:
            Suppose ``weight=0.5`` and the source wet upper-tail quantiles are
            80 (source A) and 120 (source B). The source-derived target quantile is
            ``0.5 * 80 + 0.5 * 120 = 100``. If the current morphed upper-tail
            quantile in convective wet points is 70, the raw uplift scale is
            ``100 / 70 = 1.43`` (before applying ``maximum_intensity_scale`` cap).
            If ``maximum_intensity_scale=1.5``, the scale remains 1.43. A point with
            combined local weight 0.6 then gets
            ``local_scale = 1 + 0.6 * (1.43 - 1) = 1.258`` and is multiplied by
            1.258, while points with near-zero weight are minimally changed.

        Args:
            result_data: Current output data to adjust.
            source_a_data: Source-a data array.
            source_b_data: Source-b data array.
            valid_mask: Boolean mask for points valid across all inputs.
            convective_weight: Diagnosed convective weight array.
            weight: Morphing weight in the range [0, 1].
            threshold: Wet-value threshold.
            upper_tail_quantile: High-end quantile used to compare source and current
                upper-tail intensity.
            convective_mask_threshold: Minimum convective weight needed for local
                restoration.
            maximum_intensity_scale: Largest allowed local scaling factor.
            upper_tail_intensity_quantile: Quantile used to decide where the strongest
                points lie.
            intensity_weight_width_fraction: Fraction of the local intensity centre
                used as the sigmoid width for the upper-tail weight.
            sigmoid_clip_limit: Maximum absolute value used to clip the normalised
                sigmoid input before exponentiation.

        Returns:
            Output data array after the upper-tail restoration.
        """
        # Use the source fields to estimate the strong-rain target for this transition.
        wet_a = source_a_data[valid_mask & (source_a_data > threshold)]
        wet_b = source_b_data[valid_mask & (source_b_data > threshold)]

        # Only apply the upper-tail lift where the field still looks locally
        # convective and wet enough to justify a strong-rain correction.
        convective_wet_mask = (
            valid_mask
            & (result_data > threshold)
            & (convective_weight > convective_mask_threshold)
        )

        # Skip upper-tail restoration unless both source fields contain wet
        # samples and there is at least one locally convective wet output point.
        # If any of these checks fail, there is no valid basis for a stable
        # upper-tail quantile comparison, so return the current field unchanged.
        if wet_a.size == 0 or wet_b.size == 0 or not np.any(convective_wet_mask):
            return result_data

        # Compute the high-end quantile for the wet occurrences in both source A
        # and source B. Then compute the same quantile for the Google FILM interpolated
        # field.
        target_quantile = (1.0 - weight) * np.quantile(
            wet_a, upper_tail_quantile, method="linear"
        ) + weight * np.quantile(wet_b, upper_tail_quantile, method="linear")
        current_quantile = np.quantile(
            result_data[convective_wet_mask], upper_tail_quantile, method="linear"
        )
        if current_quantile <= 0.0:
            return result_data

        # Compare the high-end source value with the current field. If the current
        # field is too weak, the ratio is greater than 1 and we need to scale it up.
        # Keep the scale factor at or above 1 and cap it to avoid excessive boosts.
        raw_scale = max(target_quantile / current_quantile, 1.0)
        raw_scale = min(raw_scale, maximum_intensity_scale)

        # Use the current field to define a high-intensity threshold for the final
        # boost. Points well above this threshold get a larger weight, while points
        # near or below it get little or no extra lift. The width is set to a
        # fraction of the threshold so the transition is smooth but still localised,
        # and the clip keeps the sigmoid input in a safe numerical range.
        intensity_centre = np.quantile(
            result_data[convective_wet_mask],
            upper_tail_intensity_quantile,
            method="linear",
        )
        intensity_width = max(
            intensity_weight_width_fraction * intensity_centre,
            np.finfo(np.float64).eps,
        )
        # The sigmoid is clipped to avoid numerical overflow in the exponentiation.
        intensity_weight = 1.0 / (
            1.0
            + np.exp(
                -np.clip(
                    (result_data - intensity_centre) / intensity_width,
                    -sigmoid_clip_limit,
                    sigmoid_clip_limit,
                )
            )
        )

        # Apply the boost as a local multiplicative factor. The current field is
        # only increased where the convective signal is present and where the
        # sigmoid weight says the pixel is in the upper tail of the wet distribution.
        # The scale is 1.0 when no uplift is needed and rises above 1.0 only where
        # the source field suggests the morphed field is too weak. We then force
        # all valid values to stay non-negative so the correction cannot create
        # negative rainfall.
        scale_weight = convective_weight * intensity_weight
        local_scale = 1.0 + scale_weight * (raw_scale - 1.0)
        result_data[valid_mask] *= local_scale[valid_mask]
        result_data[valid_mask] = np.maximum(result_data[valid_mask], 0.0)
        return result_data

    def process(
        self,
        result_cube: Cube,
        source_a: Cube,
        source_b: Cube,
        weight: float,
    ) -> Cube:
        """Suppress weak precipitation excess in a morphed field.

        The weighted mean of the two source fields is used as a smoothly varying
        guide rather than as a replacement field. Weak FILM precipitation is
        reduced more strongly than moderate or intense precipitation, helping to
        suppress broad, weak wet halos while preserving stronger, coherent
        precipitation features that FILM is intended to represent.

        Args:
            result_cube: Precipitation field produced by morphing.
            source_a: Precipitation source field at the beginning of the
                transition.
            source_b: Precipitation source field at the end of the transition.
            weight: Interpolation weight in [0, 1], where 0 corresponds to
                source A and 1 corresponds to source B.

        Returns:
            Cube containing the suppression-adjusted FILM result.

        Raises:
            ValueError: If source and result shapes are inconsistent, or if
                suppression settings fail validation.

        Notes:
            This method is intended for precipitation diagnostics only. No
            diagnostic-name validation is applied because different
            precipitation diagnostics may be supported by the same suppression
            logic.
        """
        config = self.suppression_config
        stages = self.suppression_stages
        self._validate_suppression_settings(config)

        source_a_data = np.asarray(source_a.data, dtype=np.float64)
        source_b_data = np.asarray(source_b.data, dtype=np.float64)
        result_data = np.asarray(result_cube.data, dtype=np.float64)

        if not (source_a_data.shape == source_b_data.shape == result_data.shape):
            raise ValueError(
                "result_cube, source_a and source_b must have matching shapes; "
                f"got {result_data.shape}, {source_a_data.shape} and "
                f"{source_b_data.shape}"
            )

        valid_mask = (
            np.isfinite(result_data)
            & np.isfinite(source_a_data)
            & np.isfinite(source_b_data)
        )
        if not np.any(valid_mask):
            output_cube = result_cube.copy()
            output_data = np.asarray(output_cube.data, dtype=np.float64)
            output_data[~valid_mask] = np.nan
            output_cube.data = output_data.astype(np.float32)
            return output_cube

        source_a_data = self._sanitize_valid_data(source_a_data, valid_mask)
        source_b_data = self._sanitize_valid_data(source_b_data, valid_mask)
        result_data_for_suppression = self._sanitize_valid_data(result_data, valid_mask)

        output_data = result_data.copy()
        weighted_reference = (1.0 - weight) * source_a_data + weight * source_b_data
        convective_weight = None

        if "weak_signal" in stages:
            output_data, weighted_reference = self._compute_weak_signal_suppression(
                result_data=result_data_for_suppression,
                source_a_data=source_a_data,
                source_b_data=source_b_data,
                weighted_reference=weighted_reference,
                valid_mask=valid_mask,
                weight=weight,
                threshold=config["occurrence_threshold"],
                quantile_for_centre=config["quantile_for_centre"],
                width_fraction=config["width_fraction"],
                maximum_suppression=config["maximum_suppression"],
                showery_neighbourhood_size=config["showery_neighbourhood_size"],
                showery_weight_factor=config["showery_weight_factor"],
                weakness_weight_factor=config["weakness_weight_factor"],
                sigmoid_clip_limit=config["sigmoid_clip_limit"],
            )

        if "convective" in stages or "upper_tail" in stages:
            output_data, convective_weight = self._diagnose_convective_weight(
                result_data=output_data,
                source_a_data=source_a_data,
                source_b_data=source_b_data,
                weighted_reference=weighted_reference,
                valid_mask=valid_mask,
                threshold=config["occurrence_threshold"],
                convective_neighbourhood_size=config["convective_neighbourhood_size"],
                convective_gain=config["convective_gain"],
                concentration_reference=config["concentration_reference"],
                concentration_scale=config["concentration_scale"],
            )

        if "upper_tail" in stages:
            output_data = self._restore_upper_tail(
                result_data=output_data,
                source_a_data=source_a_data,
                source_b_data=source_b_data,
                valid_mask=valid_mask,
                convective_weight=convective_weight,
                weight=weight,
                threshold=config["occurrence_threshold"],
                upper_tail_quantile=config["upper_tail_quantile"],
                convective_mask_threshold=config["convective_mask_threshold"],
                maximum_intensity_scale=config["maximum_intensity_scale"],
                upper_tail_intensity_quantile=config["upper_tail_intensity_quantile"],
                intensity_weight_width_fraction=config[
                    "intensity_weight_width_fraction"
                ],
                sigmoid_clip_limit=config["sigmoid_clip_limit"],
            )

        # Preserve invalid-input locations as NaN in the final output. Local
        # suppression diagnostics use sanitised arrays for stability, but any
        # grid point that is invalid in the inputs should remain invalid.
        output_data[~valid_mask] = np.nan

        if np.allclose(output_data, result_data, equal_nan=True):
            return result_cube.copy()

        output_cube = result_cube.copy()
        output_cube.data = output_data.astype(np.float32)
        return output_cube
