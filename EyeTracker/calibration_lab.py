import json
import math
import itertools
import numpy as np
from pathlib import Path
import os

# Import your existing agent directly!
try:
    from agent import CognitiveTMTAgent
except ImportError:
    print("Error: Could not import CognitiveTMTAgent. Make sure agent.py is in the same folder.")
    exit(1)

# --- 1. THE GROUND TRUTH METRICS ---
# These are the original human metrics we want the analyzer to perfectly rediscover
GROUND_TRUTH = {
    "A": {
        "Memory (ms)": 48,
        "Search (ms)": 988,
        "Motor (ms)": 417,
        "Total Skips": 19,
        "Saccades/Search": 214.2 / 24.0  # Converted to per-target average
    },
    "B": {
        "Memory (ms)": 78,
        "Search (ms)": 1876,
        "Motor (ms)": 510,
        "Total Skips": 61,
        "Saccades/Search": 438.5 / 24.0
    }
}

# --- 2. THE GRID SEARCH PARAMETERS ---
# The script will test every possible combination of these values
TEST_PARAMS = {
    "aoi_multipliers": [1.5, 2.0, 2.5, 3.0, 3.5],
    "min_fixations_ms": [50, 75, 100, 125, 150],
    "saccade_velocities": [0.3, 0.5, 0.7, 0.9]
}

class HeadlessAnalyzer:
    """A pure-math version of your visualizer's analysis engine, completely decoupled from Tkinter."""
    def __init__(self, aoi_mult, min_fix_ms, sac_vel):
        self.aoi_mult = aoi_mult
        self.min_fix_ms = min_fix_ms
        self.sac_vel = sac_vel
        self.canvas_size = 918 # Assuming a standard 1080p screen (1080 * 0.85)
        self.node_radius = int(self.canvas_size * 0.025)
        self.aoi_radius = self.node_radius * self.aoi_mult

    def _get_canvas_pos(self, pct_x, pct_y):
        padding = self.canvas_size * 0.08
        active = self.canvas_size - (padding * 2)
        return int((pct_x / 100.0) * active + padding), int((pct_y / 100.0) * active + padding)

    def analyze(self, jsonl_path, asc_path):
        # 1. Parse JSON for clicks and layout
        events = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): events.append(json.loads(line))
                
        layout = {t: c for t, c in zip(events[0]["targets"], events[0]["layout"])}
        clicks = [{"id": ev["target"], "time": ev["elapsed_since_start_ms"], "cx": self._get_canvas_pos(*layout[ev["target"]])[0], "cy": self._get_canvas_pos(*layout[ev["target"]])[1]} 
                  for ev in events if ev.get("event_type") == "correct_click"]

        # 2. Parse ASC for Gaze (Simplified parser for the bot's perfectly clean ASC output)
        gaze_samples = []
        with open(asc_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3 and parts[0].isdigit() and parts[1].replace('.','',1).isdigit():
                    gaze_samples.append((int(parts[0]), float(parts[1]), float(parts[2])))

        # 3. Timeline Slicer
        task_memory, task_search, task_motor, task_saccades = [], [], [], []
        task_skips = 0

        for i in range(1, len(clicks)):
            prev, curr = clicks[i-1], clicks[i]
            segment = [g for g in gaze_samples if prev["time"] <= g[0] <= curr["time"]]
            if not segment: continue

            # Working Memory
            t_leave = prev["time"]
            for g in segment:
                if math.hypot(g[1] - prev["cx"], g[2] - prev["cy"]) > self.aoi_radius:
                    t_leave = g[0]
                    break
            task_memory.append(t_leave - prev["time"])

            # Search & Motor
            t_fix_start = None
            fixation_timer, saccade_count = 0, 0
            last_g_time = t_leave

            for g in segment:
                if g[0] < t_leave: continue
                dt = g[0] - last_g_time
                if dt > 0:
                    vel = math.hypot(g[1] - segment[segment.index(g)-1][1], g[2] - segment[segment.index(g)-1][2]) / dt
                    if vel > self.sac_vel: saccade_count += 1
                last_g_time = g[0]

                if math.hypot(g[1] - curr["cx"], g[2] - curr["cy"]) <= self.aoi_radius:
                    if fixation_timer == 0: fix_start_t = g[0]
                    fixation_timer += dt
                    if fixation_timer >= self.min_fix_ms and t_fix_start is None:
                        t_fix_start = fix_start_t
                else:
                    if 0 < fixation_timer < self.min_fix_ms: task_skips += 1
                    fixation_timer = 0

            if t_fix_start is None: t_fix_start = curr["time"]
            task_search.append(t_fix_start - t_leave)
            task_motor.append(curr["time"] - t_fix_start)
            task_saccades.append(saccade_count)

        return {
            "Memory (ms)": float(np.mean(task_memory)) if task_memory else 0.0,
            "Search (ms)": float(np.mean(task_search)) if task_search else 0.0,
            "Motor (ms)": float(np.mean(task_motor)) if task_motor else 0.0,
            "Total Skips": int(task_skips),
            "Saccades/Search": float(np.mean(task_saccades)) if task_saccades else 0.0
        }

def calculate_error(measured, truth):
    """Calculates the Mean Absolute Percentage Error (MAPE) across all metrics."""
    error = 0
    for k in truth.keys():
        norm = max(truth[k], 1) # Prevent divide by zero
        error += abs(measured[k] - truth[k]) / norm
    return error

def get_latest_sim_files():
    """Finds the most recently generated JSONL and ASC files in the sim_logs folder."""
    log_dir = Path("sim_logs")
    json_files = sorted(log_dir.glob("*.jsonl"), key=os.path.getmtime, reverse=True)
    asc_files = sorted(log_dir.glob("*.asc"), key=os.path.getmtime, reverse=True)
    return json_files[0], asc_files[0]

def run_calibration():
    print("--- 🚀 Booting Cognitive Analyzer Calibration Lab ---")
    results_ledger = []
    
    for task_type in ["A", "B"]:
        print(f"\n[1/3] Generating Ground-Truth Baseline Data for TMT-{task_type}...")
        
        # Instantiate your agent and forcefully inject the ground truth metrics!
        agent = CognitiveTMTAgent(task_type=task_type, participant_id="CalibrationBot")
        agent.metrics.memory_speed = GROUND_TRUTH[task_type]["Memory (ms)"]
        agent.metrics.search_speed = GROUND_TRUTH[task_type]["Search (ms)"]
        agent.metrics.motor_speed = GROUND_TRUTH[task_type]["Motor (ms)"]
        agent.metrics.total_skips = GROUND_TRUTH[task_type]["Total Skips"]
        agent.metrics.saccades_per_target = GROUND_TRUTH[task_type]["Saccades/Search"]
        agent.run_simulation()
        
        json_path, asc_path = get_latest_sim_files()
        print(f"[2/3] Sim complete. Reading {json_path.name} and {asc_path.name}...")
        
        print(f"[3/3] Commencing Grid Search...")
        combos = list(itertools.product(TEST_PARAMS["aoi_multipliers"], TEST_PARAMS["min_fixations_ms"], TEST_PARAMS["saccade_velocities"]))
        
        best_error = float('inf')
        best_config = None
        
        for aoi, fix, vel in combos:
            analyzer = HeadlessAnalyzer(aoi, fix, vel)
            measured_metrics = analyzer.analyze(json_path, asc_path)
            error = calculate_error(measured_metrics, GROUND_TRUTH[task_type])
            
            run_data = {
                "task": task_type,
                "parameters": {"AOI_RADIUS_MULTIPLIER": aoi, "MIN_FIXATION_MS": fix, "SACCADE_VELOCITY_THRESHOLD": vel},
                "ground_truth": GROUND_TRUTH[task_type],
                "measured_results": measured_metrics,
                "error_score": error
            }
            results_ledger.append(run_data)
            
            if error < best_error:
                best_error = error
                best_config = run_data

        print(f"✅ Best Configuration for TMT-{task_type} Found!")
        print(f"   AOI: {best_config['parameters']['AOI_RADIUS_MULTIPLIER']}, FIX: {best_config['parameters']['MIN_FIXATION_MS']}ms, VEL: {best_config['parameters']['SACCADE_VELOCITY_THRESHOLD']}")
        print(f"   Error Score: {best_error:.4f}\n")

    # Sort the ledger so the best results are at the top
    results_ledger.sort(key=lambda x: x["error_score"])

    # Save to a highly readable JSON folder/file structure
    output_dir = Path("calibration_results")
    output_dir.mkdir(exist_ok=True)
    out_file = output_dir / "analyzer_calibration_report.json"
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "experiment_summary": "Analyzed synthetic bot data generated from hardcoded human metrics to find optimal analyzer constants.",
            "best_overall_runs": results_ledger[:10],
            "all_runs": results_ledger
        }, f, indent=4)
        
    print(f"🎉 Calibration Complete! Full report saved to: {out_file.resolve()}")

if __name__ == "__main__":
    run_calibration()