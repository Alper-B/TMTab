import json
import re
import math
import numpy as np
from pathlib import Path
import itertools

# We import your agent file directly! Make sure your file is named agent.py
# If it's named something else (like TMTagent.py), change this import.
try:
    import agent
except ImportError:
    print("Error: Could not import agent.py. Make sure the file name matches!")
    exit(1)

# --- THE GRID SEARCH SPACE ---
# The script will test every combination of these parameters!
PARAM_GRID = {
    "AOI_RADIUS_MULTIPLIER": [1.5, 2.0, 2.5, 3.0],
    "MIN_FIXATION_MS": [50, 100, 150],
    "SACCADE_VELOCITY_THRESHOLD": [0.3, 0.5, 0.8]
}

# Headless Canvas Configuration (Matches 1080p at 0.85 scaling)
SCREEN_W, SCREEN_H = 1920, 1080
CANVAS_SIZE = int(SCREEN_H * 0.85)
NODE_RADIUS = int(CANVAS_SIZE * 0.025)

def parse_files(json_path, asc_path):
    """Headless parser for JSONL and ASC files."""
    events, gaze_samples = [], []
    layout = {}
    
    # 1. Parse JSON
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): events.append(json.loads(line))
            
    if events:
        first = events[0]
        targets = first.get("targets", [])
        raw_layout = first.get("layout", [])
        layout = {targets[i]: raw_layout[i] for i in range(min(len(targets), len(raw_layout)))}

    # 2. Parse ASC
    sample_pattern = re.compile(r"^\s*(\d+)\s+([^\s]+)\s+([^\s]+)")
    sync_time = 0
    with open(asc_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if match := re.match(r"^MSG\s+(\d+)\s+TMT_EVENT:\s+timer_started", s):
                sync_time = int(match.group(1))
                break

    with open(asc_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if match := sample_pattern.match(line.strip()):
                ts, x, y = match.groups()
                if x != "." and y != ".":
                    gaze_samples.append((int(ts) - sync_time, float(x), float(y)))

    return events, gaze_samples, layout

def get_node_canvas_pos(pct_x, pct_y):
    padding = CANVAS_SIZE * 0.08
    active_area = CANVAS_SIZE - (padding * 2)
    return int((pct_x / 100.0) * active_area + padding), int((pct_y / 100.0) * active_area + padding)

def extract_metrics(events, gaze_samples, layout, aoi_mult, min_fix, sacc_vel):
    """The core cognitive analysis engine, running headlessly."""
    aoi_radius = NODE_RADIUS * aoi_mult
    
    # Map clicks
    clicks = []
    for ev in events:
        if ev.get("event_type") == "correct_click" and "target" in ev:
            t_id = str(ev["target"])
            if t_id in layout:
                cx, cy = get_node_canvas_pos(*layout[t_id])
                clicks.append({"id": t_id, "time": ev["elapsed_since_start_ms"], "cx": cx, "cy": cy})

    # For headless bot analysis, we assume perfect calibration (1:1 with screen space)
    # The real data should ideally be pre-calibrated or we offset it.
    offset_x = (SCREEN_W - CANVAS_SIZE) / 2
    offset_y = (SCREEN_H - CANVAS_SIZE) / 2
    calibrated_gaze = [(t, x - offset_x, y - offset_y) for t, x, y in gaze_samples]

    mem_times, search_times, motor_times, search_saccades = [], [], [], []
    skips = 0

    for i in range(1, len(clicks)):
        prev, curr = clicks[i-1], clicks[i]
        segment = [g for g in calibrated_gaze if prev["time"] <= g[0] <= curr["time"]]
        if not segment: continue

        t_leave = prev["time"]
        for g in segment:
            if math.hypot(g[1] - prev["cx"], g[2] - prev["cy"]) > aoi_radius:
                t_leave = g[0]
                break
        mem_times.append(t_leave - prev["time"])

        t_fix_start = None
        fix_timer, sacc_count = 0, 0
        last_g_time = t_leave
        
        for g in segment:
            if g[0] < t_leave: continue
            
            td = g[0] - last_g_time
            if td > 0:
                vel = math.hypot(g[1] - segment[segment.index(g)-1][1], g[2] - segment[segment.index(g)-1][2]) / td
                if vel > sacc_vel: sacc_count += 1
            last_g_time = g[0]

            if math.hypot(g[1] - curr["cx"], g[2] - curr["cy"]) <= aoi_radius:
                if fix_timer == 0: fix_start_t = g[0]
                fix_timer += td
                if fix_timer >= min_fix and t_fix_start is None: t_fix_start = fix_start_t
            else:
                if 0 < fix_timer < min_fix: skips += 1
                fix_timer = 0

        if t_fix_start is None: t_fix_start = curr["time"]

        search_times.append(t_fix_start - t_leave)
        motor_times.append(curr["time"] - t_fix_start)
        search_saccades.append(sacc_count)

    return {
        "memory_speed": float(np.mean(mem_times)) if mem_times else 0.0,
        "search_speed": float(np.mean(search_times)) if search_times else 0.0,
        "motor_speed": float(np.mean(motor_times)) if motor_times else 0.0,
        "total_skips": skips,
        "total_saccades": float(np.sum(search_saccades)) if search_saccades else 0.0
    }

def calculate_mape(real_metrics, bot_metrics):
    """Calculates the Mean Absolute Percentage Error between real and bot metrics."""
    error = 0.0
    keys = ["memory_speed", "search_speed", "motor_speed", "total_skips", "total_saccades"]
    for k in keys:
        if real_metrics[k] == 0: continue
        # How far off is the bot from the real human as a percentage?
        pct_diff = abs(real_metrics[k] - bot_metrics[k]) / real_metrics[k]
        error += pct_diff
    return error / len(keys)

# --- MONKEY PATCHING THE AGENT ---
def override_agent_metrics(self, task_type):
    """This function dynamically replaces your agent's hardcoded init."""
    with open('temp_bot_config.json', 'r') as f:
        config = json.load(f)
    
    data = config.get(task_type, config.get("A")) # Fallback to A if missing
    self.task_type = task_type
    self.memory_speed = data['memory_speed']
    self.search_speed = data['search_speed']
    self.motor_speed = data['motor_speed']
    self.total_skips = data['total_skips']
    self.total_saccades = data['total_saccades']
    self.skips_per_target = self.total_skips / 24.0
    self.saccades_per_target = self.total_saccades / 24.0

# Apply the hijack!
agent.CognitiveMetrics.__init__ = override_agent_metrics


def run_optimization(real_json_a, real_asc_a):
    print("🚀 Booting Hyperparameter Optimization Pipeline...")
    
    # Generate all combinations of parameters
    keys, values = zip(*PARAM_GRID.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Loaded {len(experiments)} experimental configurations to test.")
    
    # Load real data ONCE to save massive amounts of time
    real_ev_a, real_gaze_a, real_lay_a = parse_files(real_json_a, real_asc_a)
    
    results_ledger = []
    best_error = float('inf')
    best_params = None

    for idx, params in enumerate(experiments):
        aoi = params["AOI_RADIUS_MULTIPLIER"]
        m_fix = params["MIN_FIXATION_MS"]
        s_vel = params["SACCADE_VELOCITY_THRESHOLD"]
        
        print(f"\n--- Running Experiment {idx+1}/{len(experiments)} ---")
        print(f"Params: AOI: {aoi}x | Fixation: {m_fix}ms | Saccade Threshold: {s_vel}")
        
        # 1. Analyze Real Data with current parameters
        real_metrics_a = extract_metrics(real_ev_a, real_gaze_a, real_lay_a, aoi, m_fix, s_vel)
        
        # 2. Save Real Metrics for the Bot to read
        with open('temp_bot_config.json', 'w') as f:
            json.dump({"A": real_metrics_a}, f)
            
        # 3. Trigger the Bot Simulation
        # The bot will automatically read temp_bot_config.json due to our monkey patch
        bot = agent.CognitiveTMTAgent(task_type="A", participant_id=f"Bot_Opt_{idx}")
        bot.run_simulation()
        
        # 4. Find the newest files the bot just spit out in sim_logs
        log_dir = Path("sim_logs")
        bot_asc_files = sorted(log_dir.glob(f"Bot_Opt_{idx}_A_*.asc"), key=lambda p: p.stat().st_mtime, reverse=True)
        bot_json_files = sorted(log_dir.glob(f"Bot_Opt_{idx}_A_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not bot_asc_files or not bot_json_files:
            print("Error: Bot didn't output files correctly. Skipping.")
            continue
            
        # 5. Analyze the Bot's simulated data with the SAME parameters
        bot_ev_a, bot_gaze_a, bot_lay_a = parse_files(bot_json_files[0], bot_asc_files[0])
        bot_metrics_a = extract_metrics(bot_ev_a, bot_gaze_a, bot_lay_a, aoi, m_fix, s_vel)
        
        # 6. Calculate the Error (MAPE)
        error = calculate_mape(real_metrics_a, bot_metrics_a)
        print(f"Result Error Score: {error:.4f} (Lower is better)")
        
        results_ledger.append({
            "experiment_id": idx,
            "parameters": params,
            "real_human_metrics": real_metrics_a,
            "bot_simulated_metrics": bot_metrics_a,
            "mape_error_score": error
        })
        
        if error < best_error:
            best_error = error
            best_params = params

    # Save the final results dictionary
    with open("optimization_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_parameters": best_params,
            "lowest_error": best_error,
            "all_experiments": results_ledger
        }, f, indent=4)
        
    print("\n🎉 OPTIMIZATION COMPLETE!")
    print(f"The closest match between Human and Bot behavior happened with:")
    print(json.dumps(best_params, indent=4))
    print("Full ledger saved to 'optimization_results.json'")

if __name__ == "__main__":
    # --- PLUG IN YOUR REAL HUMAN DATA FILES HERE ---
    REAL_JSON_A = "participant_A_20260831_140357_2.jsonl"
    REAL_ASC_A = "participant_A_20260831_140357_2.asc" # Provide the corresponding ASC file path here!
    
    run_optimization(REAL_JSON_A, REAL_ASC_A)