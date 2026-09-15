import json
import itertools
import math
import re
from pathlib import Path
import numpy as np

# Import our newly refactored modular code
from agent import CognitiveTMTAgent
from visualizer import TMTSession, HeadlessCognitiveAnalyzer

def load_headless_session(json_path, asc_path, task_type="A"):
    """Fully functional headless loader that actually reads the files."""
    session = TMTSession(task_type)
    
    # 1. Parse JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): 
                    session.events.append(json.loads(line))
                    
        if session.events:
            first_event = session.events[0]
            targets = first_event.get("targets", [])
            raw_layout = first_event.get("layout", [])
            session.layout = {targets[i]: raw_layout[i] for i in range(min(len(targets), len(raw_layout)))}
            for ev in reversed(session.events):
                if ev.get("elapsed_since_start_ms") is not None:
                    session.max_time_ms = ev["elapsed_since_start_ms"]
                    break
            session.is_json_loaded = True
    except FileNotFoundError:
        print(f"Warning: JSON not found at {json_path}")

    # 2. Parse ASC
    sample_pattern = re.compile(r"^\s*(\d+)\s+([^\s]+)\s+([^\s]+)")
    msg_timer_pattern = re.compile(r"^MSG\s+(\d+)\s+TMT_EVENT:\s+timer_started")
    msg_calib_pattern = re.compile(r"^MSG\s+(\d+)\s+TMT_EVENT:\s+CALIBRATION_DOT_(\d+)_X:(\d+)_Y:(\d+)")

    sync_time, first_ts = None, None
    try:
        with open(asc_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if timer_match := msg_timer_pattern.match(s):
                    sync_time = int(timer_match.group(1))
                    break
                if not first_ts and (match := sample_pattern.match(s)):
                    first_ts = int(match.group(1))

        if sync_time is None: sync_time = first_ts if first_ts else 0

        with open(asc_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if calib_match := msg_calib_pattern.match(s):
                    ts, idx, cx, cy = int(calib_match.group(1)) - sync_time, int(calib_match.group(2)), int(calib_match.group(3)), int(calib_match.group(4))
                    session.calib_events.append((ts, idx, cx, cy))
                    continue
                if match := sample_pattern.match(s):
                    ts, x_str, y_str = match.groups()
                    if x_str == "." or y_str == ".": continue
                    try: session.gaze_samples.append((int(ts) - sync_time, float(x_str), float(y_str)))
                    except ValueError: pass
        
        if session.gaze_samples:
            session.min_time_ms = session.gaze_samples[0][0]
        session.is_asc_loaded = True
    except FileNotFoundError:
        print(f"Warning: ASC not found at {asc_path}")
        
    return session

def calculate_loss(real_metrics, sim_metrics):
    """Calculates Mean Squared Error between the human and the bot."""
    loss = 0
    keys = ["Memory (ms)", "Search (ms)", "Motor (ms)", "Total Skips", "Saccades/Search"]
    for k in keys:
        # Normalize the scale so large ms values don't overpower small skip counts
        norm_factor = max(real_metrics[k], 1.0)
        diff = (real_metrics[k] - sim_metrics[k]) / norm_factor
        loss += diff ** 2
    return loss

def run_hyperparameter_optimization():
    print("--- Starting Cognitive Parameter Optimizer ---")
    
    # 1. Define the Grid Search Space
    aoi_mults = [1.5, 2.5, 3.5]
    min_fix_times = [50, 100, 150]
    saccade_threshs = [0.3, 0.5, 0.7]
    
    # Put your real human baseline files here!
    REAL_JSON = "participant_A_20260831_140357_2.jsonl"
    REAL_ASC = "participant_A_20260831_140357_2.asc" 
    
    best_loss = float('inf')
    best_config = {}
    optimization_ledger = []

    combinations = list(itertools.product(aoi_mults, min_fix_times, saccade_threshs))
    print(f"Total configurations to test: {len(combinations)}\n")

    for aoi, fix, sac in combinations:
        print(f"Testing Config -> AOI: {aoi}, Min Fix: {fix}ms, Saccade Vel: {sac}")
        
        # Step A: Analyze REAL data with current parameters
        analyzer = HeadlessCognitiveAnalyzer(aoi, fix, sac, 918)
        real_session = load_headless_session(REAL_JSON, REAL_ASC, "A")
        real_metrics = analyzer.analyze_session(real_session)
        
        # Step B: Spin up the Bot using the real metrics
        agent = CognitiveTMTAgent(task_type="A", participant_id="OptiBot", custom_metrics=real_metrics)
        sim_json, sim_asc = agent.run_simulation()
        
        # Step C: Analyze the BOT data with the SAME parameters
        sim_session = load_headless_session(sim_json, sim_asc, "A")
        sim_metrics = analyzer.analyze_session(sim_session)
        
        # Step D: Compare them
        current_loss = calculate_loss(real_metrics, sim_metrics)
        
        result_log = {
            "parameters": {"AOI": aoi, "FIX": fix, "SAC": sac},
            "human_metrics": real_metrics,
            "bot_metrics": sim_metrics,
            "loss": current_loss,
            "files": {"json": sim_json, "asc": sim_asc}
        }
        optimization_ledger.append(result_log)
        
        if current_loss < best_loss:
            best_loss = current_loss
            best_config = result_log
            print(f"  >>> NEW BEST! Loss: {current_loss:.4f}")

    # Step E: File Cleanup (The Janitor Routine)
    print("\n--- Cleaning up suboptimal simulation files ---")
    best_json = best_config["files"]["json"]
    best_asc = best_config["files"]["asc"]
    
    files_deleted = 0
    for log in optimization_ledger:
        j_path = Path(log["files"]["json"])
        a_path = Path(log["files"]["asc"])
        
        if j_path != Path(best_json) and j_path.exists():
            j_path.unlink()
            files_deleted += 1
        if a_path != Path(best_asc) and a_path.exists():
            a_path.unlink()
            files_deleted += 1
            
    print(f"Deleted {files_deleted} discarded simulation files.")

    # Save the absolute best parameters
    with open("OPTIMIZED_COGNITIVE_PARAMS.json", "w") as f:
        json.dump(best_config, f, indent=4)
        
    print(f"\n--- Optimization Complete ---")
    print(f"Lowest Loss Achieved: {best_loss:.4f}")
    print("\n🏆 THE WINNING HYPERPARAMETERS 🏆")
    print(f"  Area of Interest Multiplier: {best_config['parameters']['AOI']}")
    print(f"  Minimum Fixation Dwell:      {best_config['parameters']['FIX']} ms")
    print(f"  Saccade Velocity Threshold:  {best_config['parameters']['SAC']} px/ms")
    print(f"\nThe champion files have been saved as:\n -> {best_json}\n -> {best_asc}")

if __name__ == "__main__":
    run_hyperparameter_optimization()