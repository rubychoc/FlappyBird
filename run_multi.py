# run_multi.py
from multi_pipe import PipeSpec, solve_pipes
import random
import tracemalloc  # NEW: for per-run peak memory

random.seed(10)

def main():
    # World bounds
    y_min, y_max = 0, 15
    nums = [random.randint(2, 13) for _ in range(100)]

    # Start

    # A course with three pipes; distances are relative re-anchored x0 for each segment
# S3a: classic sandwich causing a dead end after the middle pipe
# G2-1
# G2-1
    pipes = [
        PipeSpec(distance=5, L=3, U=6),   # interior {4,5} → greedy tends to end at 3/4
        PipeSpec(distance=5, L=6, U=9),   # interior {7,8} still reachable from 3/4, but with many clicks/branches
    ]


    # Global fix: finish pipe 2 at 6 (CLICK), then pipe 3 becomes reachable.


    NUM_RUNS = 100

    for algo in ("astar", "idastar", "ucs", "ilbfs"):
        total_cost = 0.0
        total_expanded = 0
        total_generated = 0
        total_duplicates = 0            # NEW
        total_open_peak = 0             # NEW
        avg_duplicates = 0.0
        total_time = 0.0
        avg_open_peak = 0
        max_open_peak = 0.0   # NEW
        total_peak_mem_mb = 0
        success_count = 0

        for i in range(NUM_RUNS):
            y0 = nums[i]

            tracemalloc.start()

            try:
                res = solve_pipes(
                    initial_y=y0,
                    y_min=y_min, y_max=y_max,
                    pipes=pipes,
                    algo=algo,
                )
            finally:
                # NEW: capture peak and stop tracing
                current_bytes, peak_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                # NEW: subtract the algorithm’s own seen_states peak (max over segments in this run)
                seen_peak_bytes = getattr(res, "max_seen_states_peak_bytes", 0)
                adj_peak_bytes = peak_bytes - seen_peak_bytes
                if adj_peak_bytes < 0:
                    adj_peak_bytes = 0  # guard against underflow

                peak_mb = adj_peak_bytes / (1024 * 1024)
                total_peak_mem_mb += peak_mb

            if res.found_all:
                success_count += 1
                total_cost += res.total_cost
                total_expanded += res.total_nodes_expanded
                total_generated += res.total_nodes_generated
                total_duplicates += res.total_nodes_duplicate   # NEW
                total_time += res.total_elapsed_sec
                total_open_peak += res.max_open_size 
                max_open_peak = max(max_open_peak, res.max_open_size) # NEW

        if success_count > 0:
            avg_cost = total_cost / success_count
            avg_expanded = total_expanded / success_count
            avg_generated = total_generated / success_count
            avg_time = total_time / success_count
            avg_duplicates = total_duplicates / success_count   # NEW
            avg_open_peak = total_open_peak / success_count     # NEW
            avg_peak_mem_mb = total_peak_mem_mb / success_count # NEW
        else:
            avg_cost = avg_expanded = avg_generated = avg_time = float('nan')
            avg_peak_mem_mb = float('nan')  # NEW

        print(f"[{algo.upper()}] runs={NUM_RUNS} successes={success_count}")
        print(f"  avg_total_cost       = {avg_cost:.2f}")
        print(f"  avg_nodes_expanded   = {avg_expanded:.2f}")
        print(f"  avg_nodes_generated  = {avg_generated:.2f}")
        print(f"  avg_nodes_duplicate  = {avg_duplicates:.2f}")   # NEW
        print(f"  avg_time_sec         = {avg_time:.6f}")
        # print(f"  avg_peak_mem_MiB     = {avg_peak_mem_mb:.3f}")  # NEW
        print(f"  avg_open_size_peak   = {avg_open_peak:.2f}")    # NEW
        print(f"  max open peak   = {max_open_peak:.2f}")    # NEW

        print()

if __name__ == "__main__":
    main()
