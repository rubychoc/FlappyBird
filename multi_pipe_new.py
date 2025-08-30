from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Iterable, Dict

from search_core import SearchResult
from search_core import astar_early, idastar, ilbfs, ucs
from flappy_1d import Flappy1D, BirdState, clicks_heuristic_factory

@dataclass(frozen=True)
class PipeSpec:
    distance: int   # x0 at the start of this pipe segment
    L: int          # lower edge (inclusive; at the pipe we require y in (L, U))
    U: int          # upper edge (inclusive)

@dataclass
class MultiRunResult:
    found_all: bool
    total_cost: float
    total_nodes_expanded: int
    total_nodes_generated: int
    total_nodes_duplicate: int
    max_seen_states_peak_bytes: int
    max_open_size: int
    total_elapsed_sec: float
    per_pipe: List[SearchResult]
    full_path: List[Tuple[int, BirdState, Optional[int]]]

def pick_algo(name: str) -> Callable:
    n = name.lower()
    if n in ("astar", "a*", "a-star", "a*-early"):
        return astar_early
    if n in ("idastar", "ida*", "ida"):
        return idastar
    if n in ("ilbfs", "rbfs"):
        return ilbfs
    if n in ("ucs", "uniform", "dijkstra"):
        return ucs
    raise ValueError("algo must be one of: astar | idastar | ilbfs | ucs")

def _interior_range(y_min: int, y_max: int) -> range:
    """Valid world interior: {y_min+1, ..., y_max-1}."""
    return range(y_min + 1, y_max)

def _pipe_interior(L: int, U: int) -> range:
    """
    Valid y at the pipe (x == 0) under the strict-interior rule: {L+1, ..., U-1}.
    (Change to range(L, U+1) if you later decide edges are allowed.)
    """
    return range(L + 1, U)

def _segment_min_clicks_to_end_y(
    y_start: int, y_end: int, d: int, L: int, U: int,
    y_min: int, y_max: int
) -> int | float:
    """
    Closed-form test: minimal clicks to go from start y at x=d to end y at x=-1 in d+1 steps,
    subject to y_pipe ∈ (L, U) at x=0.
    Dynamics: each of the first d steps changes y by ±1. Let c_pre = #clicks in those d steps.
      y_pipe = y_start - d + 2*c_pre.
    Final step:
      - If final action is NOCLICK (cost 0): y_end = y_pipe - 1  =>  y_pipe = y_end + 1
      - If final action is CLICK   (cost 1): y_end = y_pipe + 1  =>  y_pipe = y_end - 1
    Total clicks = c_pre (+1 if final action is CLICK).
    Feasibility conditions:
      - y_pipe ∈ (L, U)
      - c_pre integer with 0 ≤ c_pre ≤ d
      - y_end ∈ interior
      - (Optional) world interior at intermediate steps is left to the solver; here we only test kinematics.
    Returns minimal clicks if feasible, else float('inf').
    """
    if y_end not in _interior_range(y_min, y_max):
        return float('inf')

    best = float('inf')
    for last_is_click, delta_cost, y_pipe in (
        (False, 0, y_end + 1),  # NOCLICK last step
        (True,  1, y_end - 1),  # CLICK last step
    ):
        if y_pipe not in _pipe_interior(L, U):
            continue
        # c_pre = (y_pipe - (y_start - d)) / 2 must be integer and in [0, d]
        num = y_pipe - (y_start - d)
        if num % 2 != 0:
            continue
        c_pre = num // 2
        if c_pre < 0 or c_pre > d:
            continue
        clicks = c_pre + delta_cost
        if clicks < best:
            best = clicks
    return best

class _EndConstrainedProblem:
    """
    Wrapper around Flappy1D that keeps the usual goal (x == -1) AND
    restricts terminal y to a provided allowed set.
    """
    def __init__(self, base: Flappy1D, allowed_end_y: Iterable[int]) -> None:
        self._base = base
        self._allowed = set(allowed_end_y)

    def initial_state(self) -> BirdState:
        return self._base.initial_state()

    def successors(self, s: BirdState):
        return self._base.successors(s)

    def is_goal(self, s: BirdState) -> bool:
        return self._base.is_goal(s) and (s.y in self._allowed)

def _plan_end_y_sequence_dp(
    y0: int, y_min: int, y_max: int, pipes: List[PipeSpec]
) -> tuple[dict[tuple[int,int], int], dict[tuple[int,int], int | None]]:
    """
    Dynamic program over segments to compute minimal total clicks from segment i with start height y.
    dp[(i, y_start)] = min total clicks to finish from segment i ... last segment.
    next_choice[(i, y_start)] = best end y at x==-1 for segment i (or None if terminal).
    """
    Y = list(_interior_range(y_min, y_max))
    N = len(pipes)
    dp: dict[tuple[int,int], int] = {}
    nxt: dict[tuple[int,int], int | None] = {}

    # Base case: after last segment
    for y in Y:
        dp[(N, y)] = 0
        nxt[(N, y)] = None

    # Fill backwards
    for i in range(N - 1, -1, -1):
        d, L, U = pipes[i].distance, pipes[i].L, pipes[i].U
        for y_s in Y:
            best = float('inf')
            best_y_end: int | None = None
            # Enumerate candidate end heights (restrict to interior to keep it small)
            for y_e in Y:
                seg_cost = _segment_min_clicks_to_end_y(y_s, y_e, d, L, U, y_min, y_max)
                if seg_cost == float('inf'):
                    continue
                total = seg_cost + dp[(i + 1, y_e)]
                if total < best:
                    best = total
                    best_y_end = y_e
            dp[(i, y_s)] = best if best != float('inf') else float('inf')
            nxt[(i, y_s)] = best_y_end

    return dp, nxt

def solve_pipes(
    initial_y: int,
    y_min: int,
    y_max: int,
    pipes: List[PipeSpec],
    algo: str,
) -> MultiRunResult:
    """
    Global plan via DP to choose end y at each segment that minimizes total clicks across the course.
    Then execute each segment with an end-height constraint (trying alternatives with the same DP
    contribution if the first choice fails due to ordering/bounds).
    """
    algo_fn = pick_algo(algo)

    full_path: List[Tuple[int, BirdState, Optional[int]]] = []
    per: List[SearchResult] = []
    total_cost = 0.0
    total_nodes_expanded = 0
    total_nodes_generated = 0
    total_nodes_duplicate = 0
    max_seen_states_peak_bytes = 0
    max_open_size = 0
    total_elapsed_sec = 0.0

    # ---------- DP plan ----------
    dp, nxt = _plan_end_y_sequence_dp(initial_y, y_min, y_max, pipes)

    # If start is infeasible globally, early exit
    if dp[(0, initial_y)] == float('inf'):
        return MultiRunResult(
            found_all=False,
            total_cost=0.0,
            total_nodes_expanded=0,
            total_nodes_generated=0,
            total_nodes_duplicate=0,
            max_seen_states_peak_bytes=0,
            max_open_size=0,
            total_elapsed_sec=0.0,
            per_pipe=[],
            full_path=[],
        )

    # Precompute, for each segment i and y_start, a ranked list of candidate y_end
    # in increasing (segment_cost + dp[next,i+1]) to allow graceful fallback if needed.
    Y = list(_interior_range(y_min, y_max))
    ranked_end_choices: dict[tuple[int,int], list[tuple[int,int]]] = {}  # (i, y_s) -> [(total_cost, y_e), ...]
    for i, p in enumerate(pipes):
        for y_s in Y:
            choices: List[tuple[int,int]] = []
            for y_e in Y:
                seg_cost = _segment_min_clicks_to_end_y(y_s, y_e, p.distance, p.L, p.U, y_min, y_max)
                if seg_cost == float('inf'):
                    continue
                total = seg_cost + dp[(i + 1, y_e)]
                choices.append((total, y_e))
            choices.sort(key=lambda t: t[0])
            ranked_end_choices[(i, y_s)] = choices

    # ---------- Execute segments with constrained goals ----------
    y_current = initial_y
    for i, p in enumerate(pipes):
        base_problem = Flappy1D(
            y0=y_current, x0=p.distance,
            lower_edge_L=p.L, upper_edge_U=p.U,
            y_min=y_min, y_max=y_max
        )
        h = clicks_heuristic_factory(p.L)

        # Try candidate end heights from best to worse (according to DP)
        candidates = [y for _, y in ranked_end_choices[(i, y_current)]]
        if not candidates:
            # No feasible end heights from this y_start according to kinematics -> infeasible
            return MultiRunResult(
                found_all=False,
                total_cost=total_cost,
                total_nodes_expanded=total_nodes_expanded,
                total_nodes_generated=total_nodes_generated,
                total_nodes_duplicate=total_nodes_duplicate,
                max_seen_states_peak_bytes=max_seen_states_peak_bytes,
                max_open_size=max_open_size,
                total_elapsed_sec=total_elapsed_sec,
                per_pipe=per,
                full_path=full_path,
            )

        solved_this_segment = False
        for y_end in candidates:
            constrained = _EndConstrainedProblem(base_problem, [y_end])
            res = algo_fn(constrained, h)  # UCS ignores h
            # accumulate metrics regardless of success to reflect work done
            if res.open_size_peak > max_open_size:
                max_open_size = res.open_size_peak
            total_nodes_expanded += res.nodes_expanded
            total_nodes_generated += res.nodes_generated
            total_nodes_duplicate += res.duplicate_nodes
            total_elapsed_sec += res.elapsed_sec
            if res.seen_bytes_peak > max_seen_states_peak_bytes:
                max_seen_states_peak_bytes = res.seen_bytes_peak

            if res.found:
                per.append(res)
                total_cost += res.cost
                # Stitch path (skip duplicate first state from second segment on)
                for idx, (s, a) in enumerate(res.path):
                    if i > 0 and idx == 0:
                        continue
                    full_path.append((i, s, a))
                y_current = res.path[-1][0].y  # end y becomes next start
                solved_this_segment = True
                break  # move to next pipe

        if not solved_this_segment:
            # All candidate end heights failed in practice (likely due to bound-ordering);
            # report graceful failure.
            return MultiRunResult(
                found_all=False,
                total_cost=total_cost,
                total_nodes_expanded=total_nodes_expanded,
                total_nodes_generated=total_nodes_generated,
                total_nodes_duplicate=total_nodes_duplicate,
                max_seen_states_peak_bytes=max_seen_states_peak_bytes,
                max_open_size=max_open_size,
                total_elapsed_sec=total_elapsed_sec,
                per_pipe=per,
                full_path=full_path,
            )

    # All segments solved
    return MultiRunResult(
        found_all=True,
        total_cost=total_cost,
        total_nodes_expanded=total_nodes_expanded,
        total_nodes_generated=total_nodes_generated,
        total_nodes_duplicate=total_nodes_duplicate,
        max_seen_states_peak_bytes=max_seen_states_peak_bytes,
        max_open_size=max_open_size,
        total_elapsed_sec=total_elapsed_sec,
        per_pipe=per,
        full_path=full_path,
    )
