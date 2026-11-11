"""Independent 2.5D public-target city environment for Stage 1."""
from __future__ import annotations

from dataclasses import dataclass
import json
from math import hypot
from time import perf_counter

import numpy as np

from .fixed_allocator import FixedAllocator, eight_neighbor_distance_map
from .rhcr_adapted import EMCStartEstimate, ExecutionModelConsistentEstimator, ExecutionProgressStartEstimator, IndexOnlyProgressStartEstimator, IssuedPathAction, RHCRAdapted, RHCRConfig, reference_projection
from .concurrent_commit_adapter import ConcurrentCommitAdapter
from .delayed_state_channel import DelayedStateChannel, StateMessage
from .aoi_occupancy_tube import AOITubeConfig, occupancy_tube, tube_conflict_count
from .local_repair import LocalConflictRepair
from .target_pool import PublicTargetPool
from .motion_graph import legal_transition

Point = tuple[int, int]


@dataclass(frozen=True)
class LifelongConfig:
    width: int = 60
    height: int = 60
    num_drones: int = 6
    active_targets: int = 12
    episode_steps: int = 300
    num_buildings: int = 32
    building_min_size: int = 4
    building_max_size: int = 10
    building_min_height: float = 2.0
    building_max_height: float = 7.0
    flight_altitude: float = 8.0
    vertical_clearance: float = 0.5
    initial_energy: float = 600.0
    move_energy: float = 1.0
    hover_energy: float = 0.15
    energy_reserve: float = 5.0
    goal_radius: float = 1.0
    service_steps: int = 1
    seed: int = 7

    def __post_init__(self) -> None:
        if self.active_targets <= 0 or self.num_drones <= 0 or self.episode_steps <= 0:
            raise ValueError("target, UAV, and step counts must be positive")
        if self.flight_altitude < self.building_max_height + self.vertical_clearance:
            raise ValueError("flight_altitude must clear the tallest building")


class LifelongCityEnv:
    """No-sensor environment with public goals and deterministic execution."""

    _MOVES = ((0, 0), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))

    def __init__(self, config: LifelongConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.allocator = FixedAllocator()
        self.reset(config.seed)

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(self.config.seed if seed is None else seed)
        self.step_id = 0
        self.building_mask = self._generate_buildings()
        self.building_heights = self._generate_building_heights()
        self.positions = self._init_drones()
        self.trajectories = [[position] for position in self.positions]
        self.energies = np.full(self.config.num_drones, self.config.initial_energy, dtype=np.float64)
        self.assignments: dict[int, int] = {}
        self.target_pool = PublicTargetPool(self.config.active_targets, self.rng)
        self.target_pool.initialize(self._sample_free_cell, set(self.positions))
        self.total_distance = 0.0
        self.total_energy = 0.0
        self.collision_count = 0
        self.step_records: list[dict[str, int | float]] = []
        self.planning_records: list[dict[str, int | float]] = []
        self.execution_records: list[dict[str, object]] = []
        self.uav_action_records: list[dict[str, object]] = []
        self.emc_belief_event_records: list[dict[str, object]] = []
        self.last_tubes: list[list[set[Point]]] = []
        self.separation_violation_events = 0
        self.separation_violation_pair_steps = 0
        self._active_separation_pairs: set[tuple[int, int]] = set()
        self.minimum_pairwise_distance = float("inf")
        self._last_safety_interventions = {"start_occupied_hold_count": 0, "duplicate_destination_hold_count": 0, "start_occupied_hold_uav_ids": (), "duplicate_destination_hold_uav_ids": ()}
        self._assert_state()

    def _generate_buildings(self) -> np.ndarray:
        mask = np.zeros((self.config.height, self.config.width), dtype=bool)
        margin = 3
        for _ in range(self.config.num_buildings):
            width = int(self.rng.integers(self.config.building_min_size, self.config.building_max_size + 1))
            height = int(self.rng.integers(self.config.building_min_size, self.config.building_max_size + 1))
            x = int(self.rng.integers(margin, max(margin + 1, self.config.width - width - margin)))
            y = int(self.rng.integers(margin, max(margin + 1, self.config.height - height - margin)))
            mask[y:y + height, x:x + width] = True
        mask[self.config.height - 8:self.config.height, :min(self.config.width, 20)] = False
        return mask

    def _generate_building_heights(self) -> np.ndarray:
        heights = np.zeros_like(self.building_mask, dtype=np.float32)
        heights[self.building_mask] = self.rng.uniform(
            self.config.building_min_height,
            self.config.building_max_height,
            size=int(self.building_mask.sum()),
        )
        return heights

    def _sample_free_cell(self, excluded: set[Point]) -> Point:
        for _ in range(10_000):
            x = int(self.rng.integers(0, self.config.width))
            y = int(self.rng.integers(0, self.config.height))
            if not self.building_mask[y, x] and (x, y) not in excluded:
                return (x, y)
        raise RuntimeError("failed to sample a free target cell")

    def _init_drones(self) -> list[Point]:
        positions: list[Point] = []
        for drone_idx in range(self.config.num_drones):
            position = (2 + (drone_idx % 6) * 3, self.config.height - 5 - (drone_idx // 6) * 3)
            if self.building_mask[position[1], position[0]] or position in positions:
                position = self._sample_free_cell(set(positions))
            positions.append(position)
        return positions

    def _valid(self, point: Point) -> bool:
        x, y = point
        return 0 <= x < self.config.width and 0 <= y < self.config.height and not self.building_mask[y, x]

    def _target_reached(self, drone_idx: int, target) -> bool:
        return hypot(self.positions[drone_idx][0] - target.position[0],
                     self.positions[drone_idx][1] - target.position[1]) <= self.config.goal_radius

    def _allocate_idle(self, planning_positions: list[Point] | None = None) -> None:
        idle = [index for index in range(self.config.num_drones) if index not in self.assignments]
        pending = {target_id: self.target_pool.active[target_id].position for target_id in self.target_pool.pending_ids()}
        positions = self.positions if planning_positions is None else planning_positions
        for drone_idx, target_id in self.allocator.allocate(self.building_mask, positions, idle, pending,
                {key: self.step_id - self.target_pool.active[key].release_step for key in pending}):
            self.assignments[drone_idx] = target_id
            self.target_pool.assign(target_id, drone_idx, self.step_id)

    def _assignment_cost(self, planning_positions: list[Point]) -> float:
        """Cost of assigning currently idle UAVs from an explicit state view.

        This is diagnostic-only: it uses the same fixed allocator as the
        control path and does not mutate assignments or targets.
        """
        idle = [index for index in range(self.config.num_drones) if index not in self.assignments]
        pending = {target_id: self.target_pool.active[target_id].position for target_id in self.target_pool.pending_ids()}
        total = 0.0
        for drone_idx, target_id in self.allocator.allocate(self.building_mask, planning_positions, idle, pending,
                {key: self.step_id - self.target_pool.active[key].release_step for key in pending}):
            target = pending[target_id]
            distances = eight_neighbor_distance_map(self.building_mask, target)
            total += float(distances[planning_positions[drone_idx][1], planning_positions[drone_idx][0]])
        return total

    def _propose_move(self, drone_idx: int) -> Point:
        current = self.positions[drone_idx]
        target_id = self.assignments.get(drone_idx)
        if target_id is None or self.energies[drone_idx] < self.config.energy_reserve + self.config.move_energy:
            return current
        target = self.target_pool.active[target_id].position
        distances = eight_neighbor_distance_map(self.building_mask, target)
        current_distance = distances[current[1], current[0]]
        candidates: list[tuple[float, float, Point]] = []
        for dx, dy in self._MOVES:
            candidate = (current[0] + dx, current[1] + dy)
            if not self._valid(candidate):
                continue
            if not legal_transition(self.building_mask, current, candidate):
                continue
            candidates.append((float(distances[candidate[1], candidate[0]]), hypot(dx, dy), candidate))
        if not candidates:
            return current
        best_distance, _, best = min(candidates, key=lambda item: (item[0], item[1], item[2][1], item[2][0]))
        return best if best_distance <= current_distance else current

    def _resolve_moves(self, proposals: list[Point]) -> list[Point]:
        """Keep Stage-1 execution collision-free without introducing MAPF.

        A UAV may not enter any cell occupied at the start of this step.  This
        intentionally conservative rule avoids node, swap, and diagonal
        crossing conflicts while RHCR remains out of scope for Stage 1.
        """
        accepted = list(self.positions)
        start_occupied = set(self.positions)
        occupied: set[Point] = set()
        interventions = {"start_occupied_hold_count": 0, "duplicate_destination_hold_count": 0, "start_occupied_hold_uav_ids": [], "duplicate_destination_hold_uav_ids": []}
        for drone_idx, proposal in enumerate(proposals):
            current = self.positions[drone_idx]
            if not legal_transition(self.building_mask, current, proposal):
                proposal = current
            # Earlier accepted diagonal segments may cross without sharing a
            # destination. Endpoint occupancy alone does not prevent this.
            from .rhcr_adapted import _diagonals_cross
            if any(_diagonals_cross(current, proposal, self.positions[j], accepted[j])
                   for j in range(drone_idx)):
                proposal = current
            if proposal != current and proposal in start_occupied:
                proposal = current
                interventions["start_occupied_hold_count"] += 1
                interventions["start_occupied_hold_uav_ids"].append(drone_idx)
            if proposal in occupied:
                proposal = current
                interventions["duplicate_destination_hold_count"] += 1
                interventions["duplicate_destination_hold_uav_ids"].append(drone_idx)
            accepted[drone_idx] = proposal
            occupied.add(proposal)
        self._last_safety_interventions = interventions | {"start_occupied_hold_uav_ids": tuple(interventions["start_occupied_hold_uav_ids"]), "duplicate_destination_hold_uav_ids": tuple(interventions["duplicate_destination_hold_uav_ids"])}
        return accepted

    def step(self) -> None:
        self._allocate_idle()
        proposals = [self._propose_move(index) for index in range(self.config.num_drones)]
        next_positions = self._resolve_moves(proposals)
        for drone_idx, (old, new) in enumerate(zip(self.positions, next_positions)):
            distance = hypot(new[0] - old[0], new[1] - old[1])
            energy = self.config.hover_energy if distance == 0.0 else self.config.move_energy * distance
            self.energies[drone_idx] -= energy
            self.total_distance += distance
            self.total_energy += energy
            target_id = self.assignments.get(drone_idx)
            if target_id is not None:
                self.target_pool.add_path_length(target_id, distance)
        self.positions = next_positions
        self.trajectories = [path + [position] for path, position in zip(self.trajectories, self.positions)]
        self.step_id += 1
        completed = []
        for drone_idx, target_id in list(self.assignments.items()):
            target = self.target_pool.active[target_id]
            if self._target_reached(drone_idx, target):
                completed.append((drone_idx, target_id))
        for drone_idx, target_id in completed:
            self.target_pool.complete(target_id, self.step_id, self._sample_free_cell, set(self.positions))
            del self.assignments[drone_idx]
        self._assert_state()
        self.step_records.append({
            "step": self.step_id,
            "active_target_count": len(self.target_pool.active),
            "completed_targets": len(self.target_pool.completed),
            "assigned_uavs": len(self.assignments),
            "collision_count": self.collision_count,
            "total_distance": self.total_distance,
            "total_energy": self.total_energy,
        })

    def run_rhcr_adapted(self, config: RHCRConfig = RHCRConfig()) -> None:
        """Execute Stage-2 global rolling-horizon planning at zero delay."""
        planner = RHCRAdapted(config)
        while self.step_id < self.config.episode_steps:
            self._allocate_idle()
            goals = [self.target_pool.active[self.assignments[index]].position if index in self.assignments else self.positions[index] for index in range(self.config.num_drones)]
            started = perf_counter()
            paths = planner.plan(self.building_mask, self.positions, goals)
            planning_time_ms = (perf_counter() - started) * 1000.0
            self.planning_records.append({
                "step": self.step_id,
                "planning_window": config.planning_window,
                "execution_horizon": config.execution_horizon,
                "replanned_uav_count": self.config.num_drones,
                "planning_time_ms": planning_time_ms,
            })
            for offset in range(1, min(config.execution_horizon, self.config.episode_steps - self.step_id) + 1):
                self._execute_reserved_step([path[offset] for path in paths])

    def run_rhcr_with_delayed_states(self, delay_steps: int, config: RHCRConfig = RHCRConfig(), tube_config: AOITubeConfig = AOITubeConfig(), local_repair: bool = False, deviation_probability: float = 0.0, global_tube_reservation: bool = False) -> None:
        """Stage-3 execution with delayed planning-state visibility.

        The first launch state is known at mission initialization. Thereafter,
        each planner cycle records and consumes only the channel's delivered
        messages. Actual motion remains in the environment and is separately
        logged; stale plans are never used to teleport an UAV back to an old
        reported position.
        """
        channel = self._make_delayed_state_channel(delay_steps)
        channel.bootstrap(self._state_messages(timestamp=0, path_index=0))
        planner = RHCRAdapted(config)
        confirmed_paths: list[list[Point]] | None = None
        confirmed_start = 0
        while self.step_id < self.config.episode_steps:
            channel.deliver(self.step_id)
            planner_states = channel.planner_states(self.step_id)
            if not 0.0 <= deviation_probability <= 1.0:
                raise ValueError("deviation_probability must lie in [0, 1]")
            # The controller may propagate its own previously confirmed command
            # path, but it never substitutes a live environment position for a
            # delayed uplink report. In this deterministic Stage-3 model that
            # command prediction is the nominal planning state.
            expected_index = self.step_id - confirmed_start if confirmed_paths is not None else -1
            planner_positions = (
                [path[expected_index] for path in confirmed_paths]
                if deviation_probability == 0.0 and confirmed_paths is not None and 0 <= expected_index <= config.planning_window
                else [message.position for message in planner_states]
            )
            self._allocate_idle(planner_positions)
            goals = [self.target_pool.active[self.assignments[index]].position if index in self.assignments else planner_positions[index] for index in range(self.config.num_drones)]
            planning_start_error = float(np.mean([hypot(planner_positions[index][0] - self.positions[index][0], planner_positions[index][1] - self.positions[index][1]) for index in range(self.config.num_drones)]))
            started = perf_counter()
            candidate_paths = planner.plan(self.building_mask, planner_positions, goals)
            planning_time_ms = (perf_counter() - started) * 1000.0
            aoi = channel.aoi(self.step_id)
            buffers = [tube_config.buffer(value) for value in aoi]
            self.last_tubes = [occupancy_tube(self.building_mask, path, buffer) for path, buffer in zip(candidate_paths, buffers)]
            if global_tube_reservation and any(buffers):
                candidate_paths = planner.plan_with_tubes(self.building_mask, planner_positions, goals, candidate_paths, buffers)
                self.last_tubes = [occupancy_tube(self.building_mask, path, buffer) for path, buffer in zip(candidate_paths, buffers)]
            repair_components, fallback, repair_time_ms = [], False, 0.0
            closure_sets: list[tuple[int, ...]] = []
            blocker_uav_ids: list[tuple[int, ...]] = []
            hard_blockers = forced_wait_blockers = 0
            direct_conflict_uavs: set[int] = set()
            initial_component_size = 0
            original_paths = [list(path) for path in candidate_paths]
            if local_repair:
                repair = LocalConflictRepair(config).repair(self.building_mask, candidate_paths, goals, self.last_tubes, buffers)
                candidate_paths = repair.paths
                repair_components, fallback, repair_time_ms = repair.repaired_components, repair.fallback, repair.planning_time_ms
                closure_sets, blocker_uav_ids = repair.closure_sets, repair.blocker_uav_ids
                hard_blockers, forced_wait_blockers = repair.hard_infeasibility_blocker_count, repair.forced_wait_blocker_count
                direct_conflict_uavs = set().union(*repair.repaired_components) if repair.repaired_components else set()
                initial_component_size = max(repair.initial_component_sizes, default=0)
            replanned = len(set().union(*closure_sets)) if closure_sets else (self.config.num_drones if not local_repair else 0)
            nonconf = set(range(self.config.num_drones)) - direct_conflict_uavs
            preserved = sum(candidate_paths[index] == original_paths[index] for index in nonconf)
            path_updates = sum(candidate_paths[index] != original_paths[index] for index in range(self.config.num_drones))
            self.planning_records.append({"step": self.step_id, "delay_steps": delay_steps, "deviation_probability": deviation_probability, "planning_window": config.planning_window, "execution_horizon": config.execution_horizon, "replanned_uav_count": replanned, "repair_scope_ratio": replanned / self.config.num_drones, "nonconflict_path_preservation_rate": preserved / len(nonconf) if nonconf else 1.0, "path_update_count": path_updates, "planning_time_ms": planning_time_ms + repair_time_ms, "mean_planning_start_error": planning_start_error, "aoi_mean": float(np.mean(aoi)), "aoi_max": max(aoi), "tube_buffer_mean": float(np.mean(buffers)), "tube_buffer_max": max(buffers), "predicted_conflict_count": tube_conflict_count(self.last_tubes), "conflict_component_count": len(repair_components), "initial_component_size": initial_component_size, "largest_component_size": max((len(component) for component in repair_components), default=0), "final_closure_size": max((len(component) for component in closure_sets), default=0), "closure_expansion_rounds": max((len(component) for component in closure_sets), default=0) - initial_component_size if closure_sets else 0, "new_blocker_count": sum(len(item) for item in blocker_uav_ids), "blocking_uav_ids": ";".join(",".join(map(str, item)) for item in blocker_uav_ids), "hard_infeasibility_blocker_count": hard_blockers, "forced_wait_blocker_count": forced_wait_blockers, "global_fallback": int(fallback), "planner_state_max_timestamp": max(message.timestamp for message in planner_states)})
            confirmed_paths, confirmed_start = candidate_paths, self.step_id
            for _ in range(min(config.execution_horizon, self.config.episode_steps - self.step_id)):
                path_index = self.step_id + 1 - confirmed_start
                if path_index > config.planning_window:
                    # The UAV safely holds its last confirmed position until a
                    # delayed state can align a new command path.
                    self._execute_controlled_step(list(self.positions), deviation_probability)
                else:
                    tube_slice = [tube[path_index] for tube in self.last_tubes]
                    self._execute_controlled_step([path[path_index] for path in confirmed_paths], deviation_probability, tube_slice)
                channel.send(self._state_messages(timestamp=self.step_id, path_index=path_index))
                channel.deliver(self.step_id)

    def run_cc_rhcr(self, delay_steps: int, config: RHCRConfig = RHCRConfig(), deviation_probability: float = 0.25) -> None:
        """Run the PIE-inspired CC-RHCR adapted baseline.

        The controller sees only delivered status packets, public map/tasks,
        and its own committed commands.  Each two-action prefix is dispatched
        unchanged; execution outcomes are sent only through the delayed
        channel and never supplied back to the controller directly.
        """
        if config.execution_horizon != 2 or config.planning_window < 2:
            raise ValueError("CC-RHCR requires planning_window >= 2 and execution_horizon = 2")
        if not 0.0 <= deviation_probability <= 1.0:
            raise ValueError("deviation_probability must lie in [0, 1]")
        channel = self._make_delayed_state_channel(delay_steps)
        channel.bootstrap(self._state_messages(timestamp=0, path_index=0))
        adapter = ConcurrentCommitAdapter(self.config.num_drones, config)
        while self.step_id < self.config.episode_steps:
            channel.deliver(self.step_id)
            messages = channel.planner_states(self.step_id)
            adapter.synchronize(messages)
            planner_positions = list(adapter.planning_starts())
            # Allocation is exactly the existing delayed-message allocation.
            self._allocate_idle([message.position for message in messages])
            goals = [
                self.target_pool.active[self.assignments[index]].position
                if index in self.assignments else planner_positions[index]
                for index in range(self.config.num_drones)
            ]
            started = perf_counter()
            batch = adapter.plan(self.building_mask, messages, goals)
            planning_time_ms = (perf_counter() - started) * 1000.0
            # This reads live state strictly after planning for diagnostics;
            # it is not retained by the adapter/controller.
            planning_start_error = float(np.mean([
                hypot(batch.planning_starts[index][0] - self.positions[index][0], batch.planning_starts[index][1] - self.positions[index][1])
                for index in range(self.config.num_drones)
            ]))
            self.planning_records.append({
                "step": self.step_id,
                "method": "CC-RHCR",
                "delay_steps": delay_steps,
                "deviation_probability": deviation_probability,
                "planning_window": config.planning_window,
                "execution_horizon": config.execution_horizon,
                "commit_window": 2,
                "active_plan_id": batch.active_plan_id,
                "commit_generation": batch.commit_generation,
                "committed_prefix_length": len(batch.committed_prefix[0]),
                "uncommitted_suffix_length": len(batch.uncommitted_suffix[0]),
                "reanchored_uav_count": adapter.last_reanchor_count,
                "planner_state_max_timestamp": max(message.timestamp for message in messages),
                "mean_planning_start_error": planning_start_error,
                "planning_time_ms": planning_time_ms,
            })
            while adapter.controller.state.committed_actions_remaining and self.step_id < self.config.episode_steps:
                command_index, commands = adapter.dispatch_next()
                self._execute_controlled_step(list(commands), deviation_probability)
                # This producer observes physical execution, but the consumer
                # can see it only after the configured delivery delay.
                channel.send(self._state_messages(timestamp=self.step_id, path_index=command_index, path_id=batch.active_plan_id))
                channel.deliver(self.step_id)

    def run_progress_causal_diagnostic(self, delay_steps: int, mode: str, config: RHCRConfig = RHCRConfig(), deviation_probability: float = 0.25) -> None:
        """Run a frozen D0--D3 state-source decomposition without new control logic.

        D0 uses the delivered position for allocation and planning; D1 exposes
        only the planner start to truth; D2 exposes only allocation to truth;
        D3 exposes both.  All branches retain the same RHCR planner, window,
        execution cadence, assignments, map, and safety resolver.
        """
        source_modes = {
            "D0": (False, False),
            "D1": (False, True),
            "D2": (True, False),
            "D3": (True, True),
        }
        if mode not in source_modes:
            raise ValueError("mode must be one of D0, D1, D2, D3")
        if not 0.0 <= deviation_probability <= 1.0:
            raise ValueError("deviation_probability must lie in [0, 1]")
        allocation_uses_truth, planner_uses_truth = source_modes[mode]
        channel = self._make_delayed_state_channel(delay_steps)
        channel.bootstrap(self._state_messages(timestamp=0, path_index=0))
        planner = RHCRAdapted(config)
        path_registry: dict[int, list[list[Point]]] = {}
        current_path_id = -1
        current_paths: list[list[Point]] | None = None
        current_plan_executed = 0
        true_executed_indices = [0 for _ in range(self.config.num_drones)]

        while self.step_id < self.config.episode_steps:
            channel.deliver(self.step_id)
            messages = channel.planner_states(self.step_id)
            delayed_positions = [message.position for message in messages]
            true_positions = list(self.positions)
            allocation_positions = true_positions if allocation_uses_truth else delayed_positions
            planner_positions = true_positions if planner_uses_truth else delayed_positions
            assignment_cost_stale = self._assignment_cost(delayed_positions)
            assignment_cost_true = self._assignment_cost(true_positions)
            self._allocate_idle(allocation_positions)
            goals = [
                self.target_pool.active[self.assignments[index]].position
                if index in self.assignments else planner_positions[index]
                for index in range(self.config.num_drones)
            ]
            plan_start_errors = [
                hypot(planner_positions[index][0] - true_positions[index][0], planner_positions[index][1] - true_positions[index][1])
                for index in range(self.config.num_drones)
            ]
            progress_errors: list[float] = []
            obsolete_prefix_steps: list[int] = []
            for drone_idx, message in enumerate(messages):
                acknowledged = path_registry.get(message.path_id)
                age = max(0, self.step_id - message.timestamp)
                if acknowledged is None:
                    progress_errors.append(0.0)
                    obsolete_prefix_steps.append(0)
                    continue
                max_index = len(acknowledged[drone_idx]) - 1
                predicted_index = min(max_index, max(0, message.path_index + age))
                predicted = acknowledged[drone_idx][predicted_index]
                progress_errors.append(hypot(predicted[0] - true_positions[drone_idx][0], predicted[1] - true_positions[drone_idx][1]))
                obsolete_prefix_steps.append(max(0, min(age, max_index - message.path_index)))
            replaced_before_execution = int(current_path_id >= 0 and current_plan_executed == 0)
            started = perf_counter()
            candidate_paths = planner.plan(self.building_mask, planner_positions, goals)
            planning_time_ms = (perf_counter() - started) * 1000.0
            current_path_id += 1
            current_paths = candidate_paths
            path_registry[current_path_id] = candidate_paths
            current_plan_executed = 0
            self.planning_records.append({
                "step": self.step_id,
                "diagnostic_mode": mode,
                "delay_steps": delay_steps,
                "deviation_probability": deviation_probability,
                "planning_window": config.planning_window,
                "execution_horizon": config.execution_horizon,
                "replanned_uav_count": self.config.num_drones,
                "planning_time_ms": planning_time_ms,
                "planner_state_max_timestamp": max(message.timestamp for message in messages),
                "mean_plan_start_error": float(np.mean(plan_start_errors)),
                "mean_stale_position_error": float(np.mean([hypot(true_positions[index][0] - delayed_positions[index][0], true_positions[index][1] - delayed_positions[index][1]) for index in range(self.config.num_drones)])),
                "mean_progress_propagation_error": float(np.mean(progress_errors)),
                "mean_obsolete_prefix_steps": float(np.mean(obsolete_prefix_steps)),
                "plan_replaced_before_execution": replaced_before_execution,
                "assignment_cost_using_stale_state": assignment_cost_stale,
                "assignment_cost_using_true_state": assignment_cost_true,
            })
            for _ in range(min(config.execution_horizon, self.config.episode_steps - self.step_id)):
                path_index = self.step_id + 1 - (self.step_id - current_plan_executed)
                # Equivalent to 1, 2, ... within this fixed execution horizon;
                # spelling it this way makes the execution index audit explicit.
                path_index = current_plan_executed + 1
                self._execute_controlled_step(
                    [path[path_index] for path in current_paths],
                    deviation_probability,
                    execution_context={
                        "delivered_messages": messages,
                        "issued_paths": current_paths,
                        "issued_indices": [path_index for _ in range(self.config.num_drones)],
                        "true_executed_indices": true_executed_indices,
                        "issued_path_id": current_path_id,
                        "plan_start_errors": plan_start_errors,
                        "progress_errors": progress_errors,
                        "obsolete_prefix_steps": obsolete_prefix_steps,
                        "plan_replaced_before_execution": replaced_before_execution,
                        "assignment_cost_stale": assignment_cost_stale,
                        "assignment_cost_true": assignment_cost_true,
                    },
                )
                current_plan_executed += 1
                channel.send(self._state_messages(timestamp=self.step_id, path_index=path_index, path_id=current_path_id, path_indices=true_executed_indices))
                channel.deliver(self.step_id)

    def run_ep_rhcr(self, delay_steps: int, config: RHCRConfig = RHCRConfig(), deviation_probability: float = 0.25, physical_audit: bool = False) -> None:
        """EP-RHCR with only a path-progress-derived effective planner start.

        Allocation still consumes the delayed state packet exactly as the RHCR
        baseline does.  The environment is read after planning solely to log
        diagnostic error; it is never supplied to the estimator or planner.
        """
        if deviation_probability != 0.25:
            raise ValueError("EP-RHCR freezes deviation_probability at 0.25")
        channel = self._make_delayed_state_channel(delay_steps)
        channel.bootstrap(self._state_messages(timestamp=0, path_index=0))
        planner = RHCRAdapted(config)
        estimator = ExecutionProgressStartEstimator(deviation_probability)
        paths_by_id: dict[int, list[list[Point]]] = {}
        actions: list[list[IssuedPathAction]] = [[] for _ in range(self.config.num_drones)]
        reported_indices_by_path: dict[int, list[int]] = {}
        path_id = -1

        while self.step_id < self.config.episode_steps:
            channel.deliver(self.step_id)
            messages = channel.planner_states(self.step_id)
            delayed_positions = [message.position for message in messages]
            estimates = [
                estimator.estimate(message, {key: value[index] for key, value in paths_by_id.items()}, actions[index], self.step_id)
                for index, message in enumerate(messages)
            ]
            planner_positions = [estimate.position for estimate in estimates]
            # Delayed-state task allocation is intentionally identical to RHCR.
            self._allocate_idle(delayed_positions)
            goals = [
                self.target_pool.active[self.assignments[index]].position
                if index in self.assignments else planner_positions[index]
                for index in range(self.config.num_drones)
            ]
            started = perf_counter()
            candidate_paths = planner.plan(self.building_mask, planner_positions, goals)
            planning_time_ms = (perf_counter() - started) * 1000.0
            path_id += 1
            paths_by_id[path_id] = candidate_paths
            reported_indices = [0 for _ in range(self.config.num_drones)]
            reported_indices_by_path[path_id] = reported_indices
            # Diagnostic-only ground truth is intentionally computed after the
            # planner call and is never reused to form a later EP start.
            plan_start_errors = [
                hypot(planner_positions[index][0] - self.positions[index][0], planner_positions[index][1] - self.positions[index][1])
                for index in range(self.config.num_drones)
            ]
            path_index_errors = [
                abs(estimate.map_index - reported_indices_by_path.get(estimate.path_id, [messages[index].path_index] * self.config.num_drones)[index])
                for index, estimate in enumerate(estimates)
            ]
            self.planning_records.append({
                "step": self.step_id,
                "delay_steps": delay_steps,
                "deviation_probability": deviation_probability,
                "planning_window": config.planning_window,
                "execution_horizon": config.execution_horizon,
                "replanned_uav_count": self.config.num_drones,
                "planning_time_ms": planning_time_ms,
                "planner_state_max_timestamp": max(message.timestamp for message in messages),
                "mean_path_index_error": float(np.mean(path_index_errors)),
                "mean_planning_start_error": float(np.mean(plan_start_errors)),
                "mean_map_path_index": float(np.mean([estimate.map_index for estimate in estimates])),
                "mean_confirmed_path_index": float(np.mean([message.path_index for message in messages])),
                "evaluated_cross_plan_transition_count": int(sum(estimate.evaluated_cross_plan_transition_count for estimate in estimates)),
                "cross_plan_unreachable_transition_count": int(sum(estimate.cross_plan_unreachable_transition_count for estimate in estimates)),
                "evaluated_cross_plan_probability_mass": float(sum(estimate.evaluated_cross_plan_probability_mass for estimate in estimates)),
                "rejected_probability_mass": float(sum(estimate.rejected_probability_mass for estimate in estimates)),
                "cross_plan_rejection_rate": float(sum(estimate.cross_plan_unreachable_transition_count for estimate in estimates)) / max(sum(estimate.evaluated_cross_plan_transition_count for estimate in estimates), 1),
                "rejected_probability_mass_ratio": float(sum(estimate.rejected_probability_mass for estimate in estimates)) / max(sum(estimate.evaluated_cross_plan_probability_mass for estimate in estimates), 1e-12),
            })
            for path_index in range(1, min(config.execution_horizon, self.config.episode_steps - self.step_id) + 1):
                explicit_hovers = [candidate_paths[index][path_index] == candidate_paths[index][path_index - 1] for index in range(self.config.num_drones)]
                for drone_idx in range(self.config.num_drones):
                    actions[drone_idx].append(IssuedPathAction(self.step_id + 1, path_id, path_index, explicit_hovers[drone_idx]))
                self._execute_controlled_step([path[path_index] for path in candidate_paths], deviation_probability, physical_audit_context=None if not physical_audit else {"estimated_planning_starts": planner_positions, "plan_id": path_id, "path_index": path_index, "explicit_hovers": explicit_hovers})
                # This is the delayed UAV telemetry payload.  It is produced
                # after execution and reaches the estimator only through the
                # fixed-delay channel; it is not a current-step planner input.
                for drone_idx in range(self.config.num_drones):
                    if not explicit_hovers[drone_idx] and self.positions[drone_idx] == candidate_paths[drone_idx][path_index]:
                        reported_indices[drone_idx] = path_index
                channel.send(self._state_messages(timestamp=self.step_id, path_index=path_index, path_id=path_id, path_indices=reported_indices))
                channel.deliver(self.step_id)

    def run_index_only_ep_rhcr(self, delay_steps: int, config: RHCRConfig = RHCRConfig(), deviation_probability: float = 0.25) -> None:
        """Frozen Index-only EP ablation with no reachability reconstruction."""
        if deviation_probability != 0.25:
            raise ValueError("Index-only EP freezes deviation_probability at 0.25")
        channel = self._make_delayed_state_channel(delay_steps)
        channel.bootstrap(self._state_messages(timestamp=0, path_index=0))
        planner = RHCRAdapted(config)
        estimator = IndexOnlyProgressStartEstimator(deviation_probability)
        paths_by_id: dict[int, list[list[Point]]] = {}
        actions: list[list[IssuedPathAction]] = [[] for _ in range(self.config.num_drones)]
        reported_indices_by_path: dict[int, list[int]] = {}
        path_id = -1

        while self.step_id < self.config.episode_steps:
            channel.deliver(self.step_id)
            messages = channel.planner_states(self.step_id)
            delayed_positions = [message.position for message in messages]
            estimates = [
                estimator.estimate(message, {key: value[index] for key, value in paths_by_id.items()}, actions[index], self.step_id)
                for index, message in enumerate(messages)
            ]
            planner_positions = [estimate.position for estimate in estimates]
            self._allocate_idle(delayed_positions)
            goals = [
                self.target_pool.active[self.assignments[index]].position
                if index in self.assignments else planner_positions[index]
                for index in range(self.config.num_drones)
            ]
            started = perf_counter()
            candidate_paths = planner.plan(self.building_mask, planner_positions, goals)
            planning_time_ms = (perf_counter() - started) * 1000.0
            path_id += 1
            paths_by_id[path_id] = candidate_paths
            reported_indices = [0 for _ in range(self.config.num_drones)]
            reported_indices_by_path[path_id] = reported_indices
            plan_start_errors = [
                hypot(planner_positions[index][0] - self.positions[index][0], planner_positions[index][1] - self.positions[index][1])
                for index in range(self.config.num_drones)
            ]
            path_index_errors = [
                abs(estimate.map_index - reported_indices_by_path.get(estimate.path_id, [messages[index].path_index] * self.config.num_drones)[index])
                for index, estimate in enumerate(estimates)
            ]
            self.planning_records.append({
                "step": self.step_id,
                "delay_steps": delay_steps,
                "deviation_probability": deviation_probability,
                "planning_window": config.planning_window,
                "execution_horizon": config.execution_horizon,
                "replanned_uav_count": self.config.num_drones,
                "planning_time_ms": planning_time_ms,
                "planner_state_max_timestamp": max(message.timestamp for message in messages),
                "mean_path_index_error": float(np.mean(path_index_errors)),
                "mean_planning_start_error": float(np.mean(plan_start_errors)),
                "mean_map_path_index": float(np.mean([estimate.map_index for estimate in estimates])),
                "mean_confirmed_path_index": float(np.mean([message.path_index for message in messages])),
                "evaluated_cross_plan_transition_count": 0,
                "cross_plan_unreachable_transition_count": 0,
                "evaluated_cross_plan_probability_mass": 0.0,
                "rejected_probability_mass": 0.0,
                "cross_plan_rejection_rate": 0.0,
                "rejected_probability_mass_ratio": 0.0,
            })
            for path_index in range(1, min(config.execution_horizon, self.config.episode_steps - self.step_id) + 1):
                explicit_hovers = [candidate_paths[index][path_index] == candidate_paths[index][path_index - 1] for index in range(self.config.num_drones)]
                for drone_idx in range(self.config.num_drones):
                    actions[drone_idx].append(IssuedPathAction(self.step_id + 1, path_id, path_index, explicit_hovers[drone_idx]))
                self._execute_controlled_step([path[path_index] for path in candidate_paths], deviation_probability)
                for drone_idx in range(self.config.num_drones):
                    if not explicit_hovers[drone_idx] and self.positions[drone_idx] == candidate_paths[drone_idx][path_index]:
                        reported_indices[drone_idx] = path_index
                channel.send(self._state_messages(timestamp=self.step_id, path_index=path_index, path_id=path_id, path_indices=reported_indices))
                channel.deliver(self.step_id)

    def run_emc_rhcr(self, delay_steps: int, config: RHCRConfig = RHCRConfig(), deviation_probability: float = 0.25, physical_audit: bool = False, belief_support_audit: bool = False) -> None:
        """EMC-RHCR with world-coordinate reconstruction of the reference tracker."""
        if deviation_probability != 0.25:
            raise ValueError("EMC-RHCR freezes deviation_probability at 0.25")
        channel = self._make_delayed_state_channel(delay_steps)
        channel.bootstrap(self._state_messages(timestamp=0, path_index=0))
        planner = RHCRAdapted(config)
        estimator = ExecutionModelConsistentEstimator(self.building_mask, deviation_probability)
        paths_by_id: dict[int, list[list[Point]]] = {}
        actions: list[list[IssuedPathAction]] = [[] for _ in range(self.config.num_drones)]
        path_id = -1
        last_anchor_timestamps = [-1 for _ in range(self.config.num_drones)]
        last_reference_path_ids = [-1 for _ in range(self.config.num_drones)]

        while self.step_id < self.config.episode_steps:
            channel.deliver(self.step_id)
            messages = channel.planner_states(self.step_id)
            delayed_positions = [message.position for message in messages]
            estimates: list[EMCStartEstimate] = [
                estimator.estimate(message, {key: value[index] for key, value in paths_by_id.items()}, actions[index], self.step_id)
                for index, message in enumerate(messages)
            ]
            audit_beliefs = [dict(estimate.distribution) for estimate in estimates]
            first_after_anchor = [message.timestamp > last_anchor_timestamps[index] for index, message in enumerate(messages)]
            last_anchor_timestamps = [message.timestamp for message in messages]
            planner_positions = [estimate.position for estimate in estimates]
            # Allocation remains exactly the delayed-state RHCR allocation.
            self._allocate_idle(delayed_positions)
            goals = [
                self.target_pool.active[self.assignments[index]].position
                if index in self.assignments else planner_positions[index]
                for index in range(self.config.num_drones)
            ]
            started = perf_counter()
            candidate_paths = planner.plan(self.building_mask, planner_positions, goals)
            planning_time_ms = (perf_counter() - started) * 1000.0
            path_id += 1
            paths_by_id[path_id] = candidate_paths
            # These use true state only after the deployed planner has already
            # received its start positions and produced a plan.
            map_errors = [hypot(estimate.position[0] - self.positions[index][0], estimate.position[1] - self.positions[index][1]) for index, estimate in enumerate(estimates)]
            exact_accuracy = float(np.mean([error == 0.0 for error in map_errors]))
            true_state_in_support_rate = float(np.mean([dict(estimate.distribution).get(self.positions[index], 0.0) > 0.0 for index, estimate in enumerate(estimates)]))
            true_nll = float(np.mean([-np.log(max(dict(estimate.distribution).get(self.positions[index], 0.0), 1e-12)) for index, estimate in enumerate(estimates)]))
            support_sizes = [len(estimate.distribution) for estimate in estimates]
            entropies = [-sum(probability * np.log(probability) for _, probability in estimate.distribution if probability > 0.0) for estimate in estimates]
            self.planning_records.append({
                "step": self.step_id,
                "delay_steps": delay_steps,
                "deviation_probability": deviation_probability,
                "planning_window": config.planning_window,
                "execution_horizon": config.execution_horizon,
                "replanned_uav_count": self.config.num_drones,
                "planning_time_ms": planning_time_ms,
                "planner_state_max_timestamp": max(message.timestamp for message in messages),
                "mean_map_position_error": float(np.mean(map_errors)),
                "p90_map_position_error": float(np.percentile(map_errors, 90)),
                "exact_position_accuracy": exact_accuracy,
                "true_state_in_support_rate": true_state_in_support_rate,
                "true_state_nll": true_nll,
                "mean_belief_support_size": float(np.mean(support_sizes)),
                "mean_belief_entropy": float(np.mean(entropies)),
                "reference_projection_rate": float(np.mean([estimate.reference_projection_rate for estimate in estimates])),
                "mean_reference_gap": float(np.mean([estimate.mean_reference_gap for estimate in estimates])),
                "mean_remaining_gap_after_projection": float(np.mean([estimate.mean_remaining_gap_after_projection for estimate in estimates])),
                "belief_probability_mass_min": float(min(sum(probability for _, probability in estimate.distribution) for estimate in estimates)),
                "belief_probability_mass_max": float(max(sum(probability for _, probability in estimate.distribution) for estimate in estimates)),
            })
            reported_indices = [0 for _ in range(self.config.num_drones)]
            for path_index in range(1, min(config.execution_horizon, self.config.episode_steps - self.step_id) + 1):
                explicit_hovers = [candidate_paths[index][path_index] == candidate_paths[index][path_index - 1] for index in range(self.config.num_drones)]
                for drone_idx in range(self.config.num_drones):
                    actions[drone_idx].append(IssuedPathAction(self.step_id + 1, path_id, path_index, explicit_hovers[drone_idx]))
                references = [path[path_index] for path in candidate_paths]
                transitions = [estimator.transition(audit_beliefs[index], references[index], explicit_hovers[index]) for index in range(self.config.num_drones)]
                beliefs_after = [dict(transition.belief) for transition in transitions]
                cross_plan = [last_reference_path_ids[index] >= 0 and last_reference_path_ids[index] != path_id for index in range(self.config.num_drones)]
                self._execute_controlled_step(references, deviation_probability, physical_audit_context=None if not physical_audit else {"estimated_planning_starts": planner_positions, "plan_id": path_id, "path_index": path_index, "explicit_hovers": explicit_hovers}, emc_support_audit_context=None if not belief_support_audit else {"beliefs_after": beliefs_after, "references": references, "cross_plan": cross_plan, "first_after_anchor": first_after_anchor})
                audit_beliefs = beliefs_after
                last_reference_path_ids = [path_id for _ in range(self.config.num_drones)]
                first_after_anchor = [False for _ in range(self.config.num_drones)]
                for drone_idx in range(self.config.num_drones):
                    if not explicit_hovers[drone_idx] and self.positions[drone_idx] == candidate_paths[drone_idx][path_index]:
                        reported_indices[drone_idx] = path_index
                channel.send(self._state_messages(timestamp=self.step_id, path_index=path_index, path_id=path_id, path_indices=reported_indices))
                channel.deliver(self.step_id)

    def _state_messages(self, timestamp: int, path_index: int, path_id: int = -1, path_indices: list[int] | None = None) -> list[StateMessage]:
        indices = [path_index for _ in range(self.config.num_drones)] if path_indices is None else path_indices
        return [StateMessage(index, timestamp, self.positions[index], float(self.energies[index]), self.assignments.get(index), indices[index], path_id) for index in range(self.config.num_drones)]

    def _make_delayed_state_channel(self, delay_steps: int) -> DelayedStateChannel:
        """Create the deployment telemetry channel (overridden only by tape infrastructure)."""
        return DelayedStateChannel(self.config.num_drones, delay_steps)

    def _execution_deviation(self, step: int, drone_idx: int, current: Point, desired: Point, deviation_probability: float) -> bool:
        """Default frozen execution-deviation draw; tape runs override only its randomness source."""
        return desired != current and self.rng.random() < deviation_probability

    def _execute_controlled_step(self, commanded_positions: list[Point], deviation_probability: float, tube_slice: list[set[Point]] | None = None, execution_context: dict[str, object] | None = None, physical_audit_context: dict[str, object] | None = None, emc_support_audit_context: dict[str, object] | None = None, hold_mask: list[bool] | None = None) -> None:
        """Track a command safely, with seeded hold deviations for diagnosis."""
        old_positions = list(self.positions)
        energy_before = list(self.energies)
        prior_assignments = dict(self.assignments)
        target_positions_before = {
            drone_idx: self.target_pool.active[target_id].position
            for drone_idx, target_id in prior_assignments.items()
        }
        tracked: list[Point] = []
        deviations = 0
        deviation_holds: list[bool] = []
        for drone_idx, (current, command) in enumerate(zip(self.positions, commanded_positions)):
            desired = reference_projection(self.building_mask, current, command)
            if hold_mask is not None and hold_mask[drone_idx]:
                desired = current
            if self._execution_deviation(self.step_id, drone_idx, current, desired, deviation_probability):
                desired = current
                deviations += 1
                deviation_holds.append(True)
            else:
                deviation_holds.append(False)
            tracked.append(desired)
        actual = self._resolve_moves(tracked)
        tracking_error = float(np.mean([hypot(actual[index][0] - commanded_positions[index][0], actual[index][1] - commanded_positions[index][1]) for index in range(self.config.num_drones)]))
        self._execute_reserved_step(actual)
        coverage = None if tube_slice is None else float(np.mean([self.positions[index] in tube_slice[index] for index in range(self.config.num_drones)]))
        safety_holds = sum(actual[index] != tracked[index] for index in range(self.config.num_drones))
        interventions = self._last_safety_interventions
        self.step_records[-1].update({"commanded_deviation_count": deviations, "safety_conflict_hold_count": safety_holds, "start_occupied_hold_count": interventions["start_occupied_hold_count"], "duplicate_destination_hold_count": interventions["duplicate_destination_hold_count"], "mean_command_tracking_error": tracking_error, "outside_nominal_occupancy_rate": float(np.mean([self.positions[index] != commanded_positions[index] for index in range(self.config.num_drones)])), "tube_coverage_rate": coverage, "mean_tube_size": None if tube_slice is None else float(np.mean([len(cells) for cells in tube_slice]))})
        if execution_context is not None:
            messages = execution_context["delivered_messages"]
            issued_paths = execution_context["issued_paths"]
            issued_indices = execution_context["issued_indices"]
            true_indices = execution_context["true_executed_indices"]
            for drone_idx in range(self.config.num_drones):
                if actual[drone_idx] == commanded_positions[drone_idx]:
                    true_indices[drone_idx] = issued_indices[drone_idx]
                if prior_assignments.get(drone_idx) is None:
                    outcome = "idle_hold"
                elif deviation_holds[drone_idx]:
                    outcome = "random_deviation_hold"
                elif actual[drone_idx] != tracked[drone_idx]:
                    outcome = "safety_conflict_hold"
                elif actual[drone_idx] == old_positions[drone_idx] and drone_idx not in self.assignments:
                    outcome = "goal_service_hold"
                else:
                    outcome = "normal_move"
                message = messages[drone_idx]
                self.execution_records.append({
                    "step": self.step_id,
                    "drone_id": drone_idx,
                    "true_position": json.dumps(self.positions[drone_idx]),
                    "last_delivered_position": json.dumps(message.position),
                    "last_delivered_timestamp": message.timestamp,
                    "last_acknowledged_path_id": message.path_id,
                    "last_acknowledged_path_index": message.path_index,
                    "issued_path_id": execution_context["issued_path_id"],
                    "issued_path": json.dumps(issued_paths[drone_idx]),
                    "predicted_progress_index": issued_indices[drone_idx],
                    "true_executed_path_index": true_indices[drone_idx],
                    "commanded_action": json.dumps(commanded_positions[drone_idx]),
                    "executed_action": json.dumps(actual[drone_idx]),
                    "execution_outcome": outcome,
                    "plan_start_error": execution_context["plan_start_errors"][drone_idx],
                    "stale_position_error": hypot(old_positions[drone_idx][0] - message.position[0], old_positions[drone_idx][1] - message.position[1]),
                    "progress_propagation_error": execution_context["progress_errors"][drone_idx],
                    "obsolete_prefix_steps": execution_context["obsolete_prefix_steps"][drone_idx],
                    "plan_replaced_before_execution": execution_context["plan_replaced_before_execution"],
                    "assignment_cost_using_stale_state": execution_context["assignment_cost_stale"],
                    "assignment_cost_using_true_state": execution_context["assignment_cost_true"],
                })
        if physical_audit_context is not None:
            estimated_starts = physical_audit_context["estimated_planning_starts"]
            path_id = physical_audit_context["plan_id"]
            path_index = physical_audit_context["path_index"]
            explicit_hovers = physical_audit_context["explicit_hovers"]
            for drone_idx in range(self.config.num_drones):
                start, command, end = old_positions[drone_idx], commanded_positions[drone_idx], self.positions[drone_idx]
                command_dx, command_dy = command[0] - start[0], command[1] - start[1]
                actual_dx, actual_dy = end[0] - start[0], end[1] - start[1]
                command_reachable = max(abs(command_dx), abs(command_dy)) <= 1 and self._valid(command)
                actual_reachable = max(abs(actual_dx), abs(actual_dy)) <= 1 and self._valid(end)
                if prior_assignments.get(drone_idx) is None:
                    outcome = "idle_hold"
                elif deviation_holds[drone_idx]:
                    outcome = "random_deviation_hold"
                elif actual[drone_idx] != tracked[drone_idx]:
                    outcome = "safety_conflict_hold"
                elif explicit_hovers[drone_idx]:
                    outcome = "explicit_hover"
                else:
                    outcome = "normal_move"
                target_id = prior_assignments.get(drone_idx)
                completed = target_id is not None and any(record.target_id == target_id and record.completion_step == self.step_id for record in self.target_pool.completed)
                completed_distance = None if not completed else hypot(end[0] - target_positions_before[drone_idx][0], end[1] - target_positions_before[drone_idx][1])
                in_bounds = 0 <= end[0] < self.config.width and 0 <= end[1] < self.config.height
                self.uav_action_records.append({
                    "step": self.step_id,
                    "drone_id": drone_idx,
                    "true_start_position": json.dumps(start),
                    "estimated_planning_start": json.dumps(estimated_starts[drone_idx]),
                    "planned_next_waypoint": json.dumps(command),
                    "true_end_position": json.dumps(end),
                    "plan_id": path_id,
                    "path_index": path_index,
                    "action_type": "explicit_hover" if explicit_hovers[drone_idx] else "move",
                    "execution_outcome": outcome,
                    "commanded_displacement": json.dumps((command_dx, command_dy)),
                    "actual_displacement": json.dumps((actual_dx, actual_dy)),
                    "is_command_single_step_reachable": int(command_reachable),
                    "is_actual_transition_single_step_reachable": int(actual_reachable),
                    "crosses_obstacle": int(in_bounds and bool(self.building_mask[end[1], end[0]])),
                    "crosses_boundary": int(not in_bounds),
                    "target_completed_this_step": int(completed),
                    "distance_to_completed_target": completed_distance,
                    "actual_distance": hypot(actual_dx, actual_dy),
                    "energy_delta": float(energy_before[drone_idx] - self.energies[drone_idx]),
                })
        if emc_support_audit_context is not None:
            beliefs = emc_support_audit_context["beliefs_after"]
            references = emc_support_audit_context["references"]
            cross_plan = emc_support_audit_context["cross_plan"]
            first_after_anchor = emc_support_audit_context["first_after_anchor"]
            start_holds = set(self._last_safety_interventions["start_occupied_hold_uav_ids"])
            duplicate_holds = set(self._last_safety_interventions["duplicate_destination_hold_uav_ids"])
            for drone_idx in range(self.config.num_drones):
                belief = beliefs[drone_idx]
                probability = float(belief.get(self.positions[drone_idx], 0.0))
                if drone_idx in start_holds:
                    event_type = "start_occupied_hold"
                elif drone_idx in duplicate_holds:
                    event_type = "duplicate_destination_hold"
                elif deviation_holds[drone_idx]:
                    event_type = "random_deviation_hold"
                elif first_after_anchor[drone_idx]:
                    event_type = "first_step_after_message_anchor"
                elif cross_plan[drone_idx]:
                    event_type = "cross_plan_projection"
                else:
                    event_type = "same_plan_move"
                self.emc_belief_event_records.append({
                    "step": self.step_id,
                    "drone_id": drone_idx,
                    "event_type": event_type,
                    "reference_waypoint": json.dumps(references[drone_idx]),
                    "true_end_position": json.dumps(self.positions[drone_idx]),
                    "one_step_kernel_support_coverage": int(self.positions[drone_idx] in belief),
                    "true_state_in_belief_support": int(self.positions[drone_idx] in belief),
                    "zero_probability_true_state": int(probability == 0.0),
                    "true_state_probability": probability,
                    "belief_support_size": len(belief),
                })

    def _execute_reserved_step(self, next_positions: list[Point]) -> None:
        """Apply one planner-reserved step and then close completed service tasks."""
        if len(next_positions) != self.config.num_drones:
            raise ValueError("planner returned an invalid number of positions")
        for drone_idx, (old, new) in enumerate(zip(self.positions, next_positions)):
            if not self._valid(new):
                raise AssertionError("planner proposed an invalid map cell")
            distance = hypot(new[0] - old[0], new[1] - old[1])
            energy = self.config.hover_energy if distance == 0.0 else self.config.move_energy * distance
            if self.energies[drone_idx] - energy < self.config.energy_reserve - 1e-9:
                raise AssertionError("planner violated energy reserve")
            self.energies[drone_idx] -= energy
            self.total_distance += distance
            self.total_energy += energy
            target_id = self.assignments.get(drone_idx)
            if target_id is not None:
                self.target_pool.add_path_length(target_id, distance)
        self.positions = next_positions
        self.trajectories = [path + [position] for path, position in zip(self.trajectories, self.positions)]
        self.step_id += 1
        completed = [(drone_idx, target_id) for drone_idx, target_id in self.assignments.items() if self._target_reached(drone_idx, self.target_pool.active[target_id])]
        for drone_idx, target_id in completed:
            self.target_pool.complete(target_id, self.step_id, self._sample_free_cell, set(self.positions))
            del self.assignments[drone_idx]
        self._assert_state()
        pairs = {(left, right) for left in range(self.config.num_drones) for right in range(left + 1, self.config.num_drones) if hypot(self.positions[left][0] - self.positions[right][0], self.positions[left][1] - self.positions[right][1]) < 1.0}
        self.separation_violation_events += len(pairs - self._active_separation_pairs)
        self.separation_violation_pair_steps += len(pairs)
        self._active_separation_pairs = pairs
        distances = [hypot(self.positions[left][0] - self.positions[right][0], self.positions[left][1] - self.positions[right][1]) for left in range(self.config.num_drones) for right in range(left + 1, self.config.num_drones)]
        self.minimum_pairwise_distance = min(self.minimum_pairwise_distance, min(distances, default=float("inf")))
        self.step_records.append({"step": self.step_id, "active_target_count": len(self.target_pool.active), "completed_targets": len(self.target_pool.completed), "assigned_uavs": len(self.assignments), "collision_count": self.collision_count, "separation_violation_pair_steps": self.separation_violation_pair_steps, "minimum_pairwise_distance": self.minimum_pairwise_distance, "total_distance": self.total_distance, "total_energy": self.total_energy})

    def run(self) -> None:
        while self.step_id < self.config.episode_steps:
            self.step()

    def summary(self) -> dict[str, int | float]:
        responses = [record.completion_step - record.release_step for record in self.target_pool.completed if record.completion_step is not None]
        unfinished = [self.step_id - record.release_step for record in self.target_pool.active.values()]
        planning = self.planning_records
        occupancy_records = [record for record in self.step_records if record.get("tube_coverage_rate") is not None]
        return {
            "seed": self.config.seed,
            "episode_steps": self.step_id,
            "uav_count": self.config.num_drones,
            "active_targets": self.config.active_targets,
            "completed_targets": len(self.target_pool.completed),
            "completed_unique_targets": len(self.target_pool.completed),
            "throughput": len(self.target_pool.completed) / max(self.step_id, 1),
            "normalized_throughput": len(self.target_pool.completed) / max(self.config.num_drones * self.step_id, 1),
            "mean_response_time": float(np.mean(responses)) if responses else 0.0,
            "p90_response_time": float(np.percentile(responses, 90)) if responses else 0.0,
            "mean_unfinished_target_age": float(np.mean(unfinished)) if unfinished else 0.0,
            "p90_unfinished_target_age": float(np.percentile(unfinished, 90)) if unfinished else 0.0,
            "collision_count": self.collision_count,
            "separation_violation_event_count": self.separation_violation_events,
            "separation_violation_rate_per_1000_pair_steps": 1000.0 * self.separation_violation_pair_steps / max(self.step_id * self.config.num_drones * (self.config.num_drones - 1) / 2, 1),
            "minimum_pairwise_distance": self.minimum_pairwise_distance,
            "total_distance": self.total_distance,
            "total_energy": self.total_energy,
            "min_energy": float(np.min(self.energies)),
            "mean_planning_time_ms": float(np.mean([record["planning_time_ms"] for record in planning])) if planning else 0.0,
            "p95_planning_time_ms": float(np.percentile([record["planning_time_ms"] for record in planning], 95)) if planning else 0.0,
            "max_planning_time_ms": float(np.max([record["planning_time_ms"] for record in planning])) if planning else 0.0,
            "mean_repair_scope_ratio": float(np.mean([record.get("repair_scope_ratio", 0.0) for record in planning])) if planning else 0.0,
            "p90_repair_scope_ratio": float(np.percentile([record.get("repair_scope_ratio", 0.0) for record in planning], 90)) if planning else 0.0,
            "full_scope_repair_rate": float(np.mean([record.get("repair_scope_ratio", 0.0) == 1.0 for record in planning])) if planning else 0.0,
            "nonconflict_path_preservation_rate": float(np.mean([record.get("nonconflict_path_preservation_rate", 1.0) for record in planning])) if planning else 1.0,
            "total_path_update_count": int(sum(record.get("path_update_count", 0) for record in planning)),
            "safety_intervention_count": int(sum(record.get("safety_conflict_hold_count", 0) for record in self.step_records)),
            "start_occupied_hold_count": int(sum(record.get("start_occupied_hold_count", 0) for record in self.step_records)),
            "duplicate_destination_hold_count": int(sum(record.get("duplicate_destination_hold_count", 0) for record in self.step_records)),
            "mean_path_update_count_per_replan": float(np.mean([record.get("path_update_count", 0) for record in planning])) if planning else 0.0,
            "mean_nominal_position_mismatch": float(np.mean([record.get("mean_command_tracking_error", 0.0) for record in self.step_records])) if self.step_records else 0.0,
            "p95_nominal_position_mismatch": float(np.percentile([record.get("mean_command_tracking_error", 0.0) for record in self.step_records], 95)) if self.step_records else 0.0,
            "outside_nominal_occupancy_rate": float(np.mean([record.get("outside_nominal_occupancy_rate", 0.0) for record in self.step_records])) if self.step_records else 0.0,
            "tube_coverage_rate": float(np.mean([record["tube_coverage_rate"] for record in occupancy_records])) if occupancy_records else 0.0,
            "mean_tube_size": float(np.mean([record["mean_tube_size"] for record in occupancy_records])) if occupancy_records else 0.0,
            "active_target_min": min(record["active_target_count"] for record in self.step_records),
            "active_target_max": max(record["active_target_count"] for record in self.step_records),
        }

    def _assert_state(self) -> None:
        if len(set(self.positions)) != len(self.positions):
            raise AssertionError("UAV collision: duplicate position")
        for x, y in self.positions:
            if not self._valid((x, y)):
                raise AssertionError("UAV entered an invalid building or map cell")
            if self.config.flight_altitude < self.building_heights[y, x] + self.config.vertical_clearance:
                raise AssertionError("UAV violates building-height clearance")
        if np.any(self.energies < self.config.energy_reserve - 1e-9):
            raise AssertionError("UAV violated energy reserve")
        self.target_pool.assert_invariants(set(self.positions))
