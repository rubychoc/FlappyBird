# multi_pipe.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional

from search_core import SearchResult
from search_core import astar_early, idastar, ilbfs
from flappy_1d import Flappy1D, BirdState, clicks_heuristic_factory

@dataclass(frozen=True)
class PipeSpec:
    distance: int   # x0 at the start of this pipe segment
    L: int          # lower edge (inclusive; we require y >= L+1 at x==0)
    U: int          # upper edge (inclusive)

@dataclass
class MultiRunResult:
    found_all: bool
    total_cost: float
    total_nodes_expanded: int
    total_nodes_generated: int
    total_elapsed_sec: float
    per_pipe: List[SearchResult]
    # combined path: (pipe_index, state, action)
    full_path: List[Tuple[int, BirdState, Optional[int]]]

def pick_algo(name: str) -> Callable:
    if name.lower() in ("astar", "a*", "a-star", "a*-early"):
        return astar_early
    if name.lower() in ("idastar", "ida*", "ida"):
        return idastar
    if name.lower() in ("ilbfs", "rbfs"):
        return ilbfs
    raise ValueError("algo must be one of: astar | idastar | ilbfs")

def solve_pipes(
    initial_y: int,
    y_min: int,
    y_max: int,
    pipes: List[PipeSpec],
    algo: str,
) -> MultiRunResult:
    """
    For each pipe i:
      - Re-anchor coordinates: start at (y_current, x0 = pipes[i].distance)
      - Create Flappy1D(probem) with that pipe's (L,U)
      - Use the same search algorithm and the per-pipe heuristic h_i built from L_i
      - Append path and aggregate metrics; stop on first failure.
    """
    algo_fn = pick_algo(algo)

    full_path: List[Tuple[int, BirdState, Optional[int]]] = []
    per: List[SearchResult] = []
    total_cost = 0.0
    total_nodes_expanded = 0
    total_nodes_generated = 0
    total_elapsed_sec = 0.0

    y_current = initial_y

    for i, p in enumerate(pipes):
        problem = Flappy1D(y0=y_current, x0=p.distance,
                           lower_edge_L=p.L, upper_edge_U=p.U,
                           y_min=y_min, y_max=y_max)
        h = clicks_heuristic_factory(p.L)

        # Run the chosen algorithm
        res = algo_fn(problem, h)  # type: ignore[arg-type]
        per.append(res)
        total_nodes_expanded += res.nodes_expanded
        total_nodes_generated += res.nodes_generated
        total_elapsed_sec += res.elapsed_sec

        if not res.found:
            # Return what we have so far; found_all=False
            return MultiRunResult(
                found_all=False,
                total_cost=total_cost,
                total_nodes_expanded=total_nodes_expanded,
                total_nodes_generated=total_nodes_generated,
                total_elapsed_sec=total_elapsed_sec,
                per_pipe=per,
                full_path=full_path,
            )

        total_cost += res.cost

        # Stitch path: skip the very first state (duplicate between segments)
        # but keep all actions. Tag with pipe index i.
        # res.path is List[(state, action)] where first action is usually None.
        for idx, (s, a) in enumerate(res.path):
            if i > 0 and idx == 0:
                continue  # avoid duplicating the prior segment's terminal state visually
            full_path.append((i, s, a))

        # Re-anchor for next pipe: carry forward the terminal y at x==-1
        y_current = res.path[-1][0].y

    return MultiRunResult(
        found_all=True,
        total_cost=total_cost,
        total_nodes_expanded=total_nodes_expanded,
        total_nodes_generated=total_nodes_generated,
        total_elapsed_sec=total_elapsed_sec,
        per_pipe=per,
        full_path=full_path,
    )
