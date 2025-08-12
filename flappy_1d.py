# flappy_1d.py  (your concrete problem + heuristic for click=1 / no-click=0)
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple, Optional

from search_core import StateLike, SearchProblem, HeuristicFn

CLICK = 1
NO_CLICK = 0

@dataclass(frozen=True)
class BirdState(StateLike):
    y: int   # vertical cell
    x: int   # distance to pipe (0 means at the pipe; goal is to get to x = -1)
    # You can embed bounds here too if they vary, or keep them global in the problem.

class Flappy1D(SearchProblem):
    def __init__(self, y0: int, x0: int, lower_edge_L: int, upper_edge_U: int,
                 y_min: int, y_max: int):
        self._start = BirdState(y0, x0)
        self.L = lower_edge_L
        self.U = upper_edge_U
        self.y_min = y_min
        self.y_max = y_max

    def initial_state(self) -> BirdState:
        return self._start

    def is_goal(self, s: BirdState) -> bool:
        # success = one step past the pipe
        return s.x == -1

    def _is_safe(self, s: BirdState) -> bool:
        # Floor/ceiling
        if s.y < self.y_min or s.y > self.y_max:
            return False

        # While x >= 0 (approach or at pipe), enforce the gap
        if s.x == 0:
            if s.y < self.L + 1:   # strictly above lower bound (your rule)
                return False
            if s.y > self.U-1:       # at/under upper bound is allowed; above U is collision
                return False
        return True

    def successors(self, s: BirdState) -> Iterable[Tuple[int, BirdState, float]]:
        # two actions: CLICK (+1y) or NO_CLICK (-1y); x always decreases by 1
        y_up = s.y + 1
        y_down = s.y - 1
        x2 = s.x - 1

        # CLICK child
        s_click = BirdState(y_up, x2)
        if self._is_safe(s_click):
            yield (CLICK, s_click, 1.0)  # click cost = 1

        # NO_CLICK child
        s_none = BirdState(y_down, x2)
        if self._is_safe(s_none):
            yield (NO_CLICK, s_none, 0.0)  # no-click cost = 0


def clicks_heuristic_factory(L: int) -> HeuristicFn:
    """
    h_clicks(y,x) = max(0, ceil(((L+1) - (y - x)) / 2))
    (Admissible under click=1, no-click=0 and your 'must be >= L+1 at the pipe' rule.)
    """
    import math
    def h(s: BirdState) -> float:
        return max(0, math.ceil(((L + 1) - (s.y - s.x)) / 2))
    return h
