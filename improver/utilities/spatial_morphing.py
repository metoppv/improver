# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of 'IMPROVER' and is released under the BSD 3-Clause license.
# See LICENSE in the root of the repository for full licensing details.
"""Plugin for spatial morphing between forecast sources using Google FILM."""

import json
from typing import Any, Dict, Optional, Tuple

import numpy as np
from iris.cube import Cube, CubeList

from improver import BasePlugin
from improver.blending.utilities import remove_blend_time, remove_deprecation_warnings
from improver.calibration.quantile_mapping import QuantileMapping
from improver.clustering.realization_clustering import RealizationSelection
from improver.utilities.temporal import (
    reset_forecast_reference_time_and_period,
    validate_cycletime_format,
)
from improver.utilities.temporal_interpolation import (
    GoogleFilmInterpolation,
    _as_tuple_if_list,
)


class SpatialMorphing(BasePlugin):
    """Spatially morph between forecast sources for a selected realization cluster.

    This plugin builds upon RealizationSelection to select realizations from multiple
    forecast sources according to cluster assignments, then applies spatial morphing
    using Google FILM to create seamless transitions between different source models.

    Unlike hard joins (RealizationSelection alone), this plugin produces spatially
    smooth blended forecasts where different sources contribute smoothly based on
    configured transition characteristics.

    Workflow:
    1. Split input cubes into forecast cubes and cluster cube (from RealizationClusterAndMatch).
    2. Validate that all forecast cubes have the same validity time.
    3. Optionally reset forecast_reference_time based on cycletime.
    4. Parse cluster mapping attributes to identify which realization from which
       forecast source corresponds to the requested cluster.
    5. Extract the selected realizations from each source.
    6. Apply Google FILM spatial morphing to create seamless blended output.

    This plugin is designed to work with output from RealizationClusterAndMatch,
    providing a more direct spatial morphing alternative to the traditional
    RealizationSelection → ForecastTrajectoryGapFiller pipeline.
    """

    def __init__(
        self,
        forecast_period: int,
        cluster_number: int,
        model_id_attr: str = "mosg__model_configuration",
        cycletime: Optional[str] = None,
        selection_attr: Optional[str] = None,
        selection_attr_value: str = "spatial_morphing",
        transitions: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
        scaling: str = "minmax",
        clipping_bounds: Optional[Tuple[float, float]] = None,
        clip_in_scaled_space: bool = True,
        clip_to_physical_bounds: bool = False,
        apply_active_fraction: bool = True,
        active_threshold: float = 0.0,
        max_batch: Optional[int] = 1,
        parallel_backend: Optional[str] = None,
        n_workers: Optional[int] = 1,
        model_loader: Any = None,
        transition_weights_scheme: str = "linear",
        apply_quantile_mapping: bool = False,
        occurrence_threshold: float = 0.0,
    ) -> None:
        """Initialise the SourceSpatialMorphing plugin.

        Args:
            forecast_period: The forecast period (in seconds) to use for interrogating
                the cluster mapping attributes in order to select the appropriate
                realizations from each forecast source.
            cluster_number: The cluster index (int) to select realizations for.
                Only this cluster will be processed; output will be a single realization
                with this index.
            model_id_attr: The name of the cube attribute used to identify the model
                source. Default: "mosg__model_configuration".
            cycletime: The forecast_reference_time on the input forecast cubes will be
                reset to this value. The forecast periods will be adjusted accordingly
                with the validity times kept fixed. cycletime should be provided in the
                format YYYYMMDDTHHMMZ (e.g., 20240101T0000Z). If not provided, the
                forecast_reference_time on the input cubes will be left unchanged.
            selection_attr: Optional name of a cube attribute to add to the output
                to identify that these realizations were selected using this plugin.
                If not provided (None), no attribute is added. Example:
                "realization_selection_method".
            selection_attr_value: The value (e.g. a description of the selection
                method) to assign to the selection_attr attribute. Default is
                "spatial_morphing". Only used if selection_attr is provided.
            transitions: Optional explicit transition dictionary defining source
                pairs and their transition bounds. Expected form is either a
                dictionary containing a "transitions" list, or a list of
                transition dictionaries with keys "source_a", "source_b",
                "start_forecast_period_minutes", and
                "end_forecast_period_minutes".
            model_path: Path to TensorFlow Hub module for Google FILM model.
                Required if spatial morphing between different sources is performed.
            scaling: Scaling method for FILM interpolation: "log10" or "minmax".
                Default: "minmax".
            clipping_bounds: Optional (min, max) bounds for clipping interpolated data.
            clip_in_scaled_space: If True, clipping applied before reverse scaling.
                Default: True.
            clip_to_physical_bounds: If True, clipping applied after reverse scaling.
                Default: False.
            apply_active_fraction: If True, adjust the morphed field so the fraction
                of active pixels matches the source-weighted active-pixel fraction.
                Default: True.
            active_threshold: Threshold defining an active pixel. Default: 0.0.
            max_batch: Maximum batch size for FILM inference. Default: 1.
            parallel_backend: Parallelization backend ("loky") or None for serial.
                Default: None.
            n_workers: Number of workers for parallel processing. Default: 1.
            model_loader: Optional callable to load the TensorFlow model.
            transition_weights_scheme: Scheme for computing transition weights:
                "linear" or "smoothstep". Default: "linear".
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
        self.apply_active_fraction = apply_active_fraction
        self.active_threshold = active_threshold
        self.max_batch = max_batch
        self.parallel_backend = parallel_backend
        self.n_workers = n_workers
        self.model_loader = model_loader
        self.transition_weights_scheme = transition_weights_scheme
        self.occurrence_threshold = occurrence_threshold
        self.apply_quantile_mapping = apply_quantile_mapping

        if self.transition_weights_scheme not in {"linear", "smoothstep"}:
            raise ValueError(
                "transition_weights_scheme must be 'linear' or 'smoothstep'"
            )

        # Create RealizationSelection helper for accessing cluster mapping methods
        self._selection_helper = RealizationSelection(
            forecast_period=forecast_period,
            model_id_attr=model_id_attr,
            cycletime=cycletime,
            selection_attr=selection_attr,
            selection_attr_value=selection_attr_value,
        )

    def _parse_transitions(
        self, transitions: Optional[Dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Validate and normalise explicit transition definitions.

        The input may be a dictionary with a top-level "transitions" list, a list of
        transition dictionaries, or None.
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
            raise ValueError(
                "transitions must be a dictionary containing a 'transitions' list or a list of transition dictionaries"
            )

        parsed_transitions: list[dict[str, Any]] = []
        for transition in transition_list:
            if not isinstance(transition, dict):
                raise ValueError("Each transition must be a dictionary")

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
                    "Transition start/end forecast periods must be positive integers with start < end"
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
        selected_source_name,
        forecast_period,
        active_transitions: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
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
        available_source_names: Optional[set[str]],
    ) -> Optional[dict[str, Any]]:
        """Return the first active transition compatible with the source set."""

        def _matches(
            transition: dict[str, Any],
            source_tag: str,
            other_source_tag: str,
        ) -> bool:
            return transition[source_tag] == selected_source_name and (
                available_source_names is None
                or transition[other_source_tag] in available_source_names
            )

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

    def _find_active_transition(
        self,
        forecast_period: int,
        selected_source_name: Optional[str] = None,
        available_source_names: Optional[set[str]] = None,
    ) -> Optional[dict[str, Any]]:
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

        raise ValueError(
            "No transition matches forecast_period="
            f"{forecast_period} for selected source {selected_source_name!r}"
        )

    @staticmethod
    def _calculate_transition_weight(
        forecast_period: int,
        start_forecast_period_seconds: int,
        end_forecast_period_seconds: int,
    ) -> float:
        """Calculate the interpolation weight between explicit transition bounds."""
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

        # Call FILM with weight as time_fraction
        result_cubes = interpolator.process(cube_a, cube_b, template)

        if len(result_cubes) == 0:
            raise RuntimeError("Google FILM interpolation returned no results")

        return result_cubes[0]

    def _select_single_source_cube(
        self,
        source_name: str,
        realization_index: Optional[int],
        forecast_cubes: CubeList,
    ) -> Optional[Cube]:
        """Select a single source cube for a given realization index.

        Args:
            source_name: Source model identifier from model_id_attr.
            realization_index: Realization index to extract.
            forecast_cubes: Input forecast cubes.

        Returns:
            Selected cube, or None if source/realization cannot be extracted.
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

    def _diagnose_realization_for_source(
        self,
        source_name: str,
        cluster_number: int,
        target_period: int,
        secondary_map: Optional[dict[str, dict[str, list[dict[str, list[int]]]]]],
        primary_map: dict[str, int],
        cluster_cube: Cube,
        full_cluster_to_selection: dict[int, tuple[str, int]],
    ) -> Optional[int]:
        """Diagnose the realization index for a source at/near a target period."""
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

    @staticmethod
    def match_active_fraction(
        film_data: np.ndarray,
        source_a: np.ndarray,
        source_b: np.ndarray,
        weight: float,
        active_threshold: float = 0.0,
    ) -> np.ndarray:
        """
        Adjust FILM output so that the fraction of active pixels matches a
        weighted combination of the source active-pixel fractions.

        Args:
            film_data: Output from Google FILM.
            source_a: Source A field.
            source_b: Source B field.
            weight: Morphing weight (0=source A, 1=source B).
            active_threshold: Threshold defining an active pixel.

        Returns:
            Adjusted FILM field.
        """
        # Active fractions of the input fields
        active_fraction_a = np.mean(source_a > active_threshold)
        active_fraction_b = np.mean(source_b > active_threshold)

        # Target active fraction
        target_active_fraction = (
            1.0 - weight
        ) * active_fraction_a + weight * active_fraction_b

        # Number of active pixels desired
        n_pixels = film_data.size
        n_active_target = round(target_active_fraction * n_pixels)

        # Degenerate cases
        if n_active_target <= 0:
            return np.zeros_like(film_data)

        if n_active_target >= n_pixels:
            return film_data.copy()

        # Find threshold that retains exactly the desired number
        threshold = np.partition(
            film_data.ravel(),
            n_pixels - n_active_target,
        )[n_pixels - n_active_target]

        result = film_data.copy()
        result[result < threshold] = 0.0
        return result

    def apply_quantile_mapping_to_morphed(
        self, result_cube: Cube, source_a: Cube, source_b: Cube, weight: float
    ) -> Cube:
        """
        Apply quantile mapping to the result cube based on source cubes and weight.

        Args:
            result_cube: Cube to be adjusted.
            source_a: Source A cube.
            source_b: Source B cube.
            weight: Morphing weight (0=source A, 1=source B).

        Returns:
            Adjusted result cube.
        """
        weighted_source_cube = result_cube.copy()
        weighted_source_cube.data = (
            1.0 - weight
        ) * source_a.data + weight * source_b.data

        result = QuantileMapping(
            occurrence_threshold=self.occurrence_threshold
        ).process(result_cube, weighted_source_cube)
        return result

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
            ValueError: If cluster cube not found, forecast cubes missing, or
                if selected realization cannot be extracted.
            RuntimeError: If Google FILM morphing fails.
        """
        # Step 1: Flatten input then split forecast and cluster cubes.
        if len(cubes) == 1 and isinstance(cubes[0], CubeList):
            cubes = tuple(cubes[0])

        forecast_cubes, cluster_cube = (
            self._selection_helper.split_cubes_forecast_and_cluster(cubes)
        )

        # Step 2: Validate all forecast cubes have same validity time
        self._selection_helper.validate_common_validity_time(forecast_cubes)

        # Step 3: Reset forecast reference time if cycletime provided
        if self.cycletime is not None:
            for cube in forecast_cubes:
                reset_forecast_reference_time_and_period(cube, self.cycletime)

        # Step 4: Parse cluster mapping attributes
        primary_map, secondary_map = self._selection_helper.parse_mapping_attributes(
            cluster_cube
        )

        # Step 5: Find nearest secondary mapping forecast period
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

        # Step 6: Build cluster-to-selection mapping (only for requested cluster)
        full_cluster_to_selection = self._selection_helper.build_cluster_to_selection(
            nearest_fp, use_secondary, secondary_map, primary_map, cluster_cube
        )

        # Only process the requested cluster
        if self.cluster_number not in full_cluster_to_selection:
            raise ValueError(
                f"Cluster number {self.cluster_number} not found in cluster mapping."
            )

        cluster_to_selection = {
            self.cluster_number: full_cluster_to_selection[self.cluster_number]
        }
        # Step 7: Select the source/realization dictated by cluster mapping for this
        # forecast period. This is the baseline output when no transition morphing is
        # required.
        selected_cubes = self._selection_helper.select_realizations_for_clusters(
            cluster_to_selection, forecast_cubes
        )
        if len(selected_cubes) == 0:
            raise RuntimeError(
                f"No realization selected for cluster {self.cluster_number}"
            )
        result_cube = selected_cubes[0]
        available_source_names = {
            cube.attributes.get(self.model_id_attr)
            for cube in forecast_cubes
            if cube.attributes.get(self.model_id_attr) is not None
        }
        # Step 8: Apply explicit transition definitions.
        selected_source_name = result_cube.attributes.get(self.model_id_attr)
        active_transition = self._find_active_transition(
            self.forecast_period,
            selected_source_name=selected_source_name,
            available_source_names=available_source_names,
        )

        if active_transition is not None:
            start_forecast_period_seconds = active_transition[
                "start_forecast_period_seconds"
            ]
            end_forecast_period_seconds = active_transition[
                "end_forecast_period_seconds"
            ]
            source_a = active_transition["source_a"]
            source_b = active_transition["source_b"]

            source_a_realization = self._diagnose_realization_for_source(
                source_name=source_a,
                cluster_number=self.cluster_number,
                target_period=start_forecast_period_seconds,
                secondary_map=secondary_map,
                primary_map=primary_map,
                cluster_cube=cluster_cube,
                full_cluster_to_selection=full_cluster_to_selection,
            )
            source_b_realization = self._diagnose_realization_for_source(
                source_name=source_b,
                cluster_number=self.cluster_number,
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

            if cube_a is not None and cube_b is not None:
                weight = self._calculate_transition_weight(
                    self.forecast_period,
                    start_forecast_period_seconds,
                    end_forecast_period_seconds,
                )

                # Apply smoothstep to weight for smoother transition
                if self.transition_weights_scheme == "smoothstep":
                    weight = weight * weight * (3.0 - 2.0 * weight)

                if weight <= 0.0:
                    result_cube = cube_a
                elif weight >= 1.0:
                    result_cube = cube_b
                else:
                    result_cube = self._call_google_film_for_morphing(
                        cube_a,
                        cube_b,
                        weight,
                    )
            if self.apply_active_fraction and cube_a is not None and cube_b is not None:
                result_cube.data = self.match_active_fraction(
                    result_cube.data,
                    cube_a.data,
                    cube_b.data,
                    weight=weight,
                    active_threshold=self.active_threshold,
                )

            if (
                self.apply_quantile_mapping
                and cube_a is not None
                and cube_b is not None
                and weight > 0.0
                and weight < 1.0
            ):
                result_cube = self.apply_quantile_mapping_to_morphed(
                    result_cube,
                    cube_a,
                    cube_b,
                    weight=weight,
                )

        # Remove blend time and sanitise forecast_reference_time attributes to
        # support merging later.
        result_cube = remove_blend_time(result_cube)
        result_cube = remove_deprecation_warnings(result_cube)

        # Ensure the output realization coordinate matches the requested cluster.
        if result_cube.coords("realization"):
            result_cube.coord("realization").points = [self.cluster_number]
            result_cube.coord("realization").units = "1"

        result_cube.attributes.pop(self.model_id_attr, None)

        # Step 9: Add selection attribute if requested
        if self.selection_attr is not None:
            result_cube.attributes[self.selection_attr] = self.selection_attr_value

        return result_cube
