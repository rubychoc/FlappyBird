# run_multi.py
from multi_pipe import PipeSpec, solve_pipes

def main():
    # World bounds
    y_min, y_max = 0, 6

    # Start
    y0 = 3

    # A course with three pipes; distances are relative re-anchored x0 for each segment
    pipes = [
        PipeSpec(distance=12, L=4, U=8),
        PipeSpec(distance=10, L=3, U=7),
        PipeSpec(distance=14, L=5, U=9),
    ]

    for algo in ("astar", "idastar", "ilbfs"):
        res = solve_pipes(
            initial_y=y0,
            y_min=y_min, y_max=y_max,
            pipes=pipes,
            algo=algo,
        )

        print(f"[{algo.upper()}] found_all={res.found_all} "
              f"total_cost={res.total_cost:.0f} "
              f"expanded={res.total_nodes_expanded} "
              f"generated={res.total_nodes_generated} "
              f"time={res.total_elapsed_sec:.6f}s")

        # Pretty print the combined path
        if res.full_path:
            print("Combined path:")
            for (pi, s, a) in res.full_path:
                action_str = "CLICK" if a == 1 else ("NO CLICK" if a == 0 else "START")
                print(f"  pipe#{pi}: {s} via {action_str}")
        print()

if __name__ == "__main__":
    main()
