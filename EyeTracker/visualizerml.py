import json
import re
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from pathlib import Path
import numpy as np
import math
import threading

# Import the biological agent for ML simulations
try:
    from agent import CognitiveTMTAgent
except ImportError:
    print("Warning: agent.py not found. ML Optimization will not work.")

# --- BIOLOGICAL COGNITIVE CONSTANTS ---
AOI_ANGLE_DEG = 2.0                 # Foveal vision (High-acuity fixation)
PERIPHERAL_ANGLE_DEG = 5.0          # Parafoveal vision (Peripheral memory encoding)
MIN_FIXATION_MS = 100               
SACCADE_VELOCITY_DEG_MS = 0.03      

class TMTSession:
    def __init__(self, task_type):
        self.task_type = task_type
        self.events, self.gaze_samples, self.calib_events, self.calib_gaze_matches = [], [], [], []
        self.layout = {}
        self.max_time_ms, self.min_time_ms = 0, 0 
        self.is_json_loaded, self.is_asc_loaded = False, False

        self.use_auto_calib = False
        self.calib_coef_x, self.calib_coef_y = [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0] 
        self.gaze_offset_x, self.gaze_offset_y = 0.0, 0.0
        self.gaze_scale_x, self.gaze_scale_y = 1.0, 1.0
        self.mouse_to_pct_m_x, self.mouse_to_pct_b_x = 1.0, 0.0
        self.mouse_to_pct_m_y, self.mouse_to_pct_b_y = 1.0, 0.0
        
        self.screen_dist_mm = 600.0  
        self.pixel_pitch_mm = 0.27   

    def pixels_to_degrees(self, px):
        if self.screen_dist_mm <= 0 or self.pixel_pitch_mm <= 0: return 0.0
        physical_size_mm = px * self.pixel_pitch_mm
        return math.degrees(2 * math.atan(physical_size_mm / (2 * self.screen_dist_mm)))

    def degrees_to_pixels(self, deg):
        if self.screen_dist_mm <= 0 or self.pixel_pitch_mm <= 0: return 0.0
        physical_size_mm = 2 * self.screen_dist_mm * math.tan(math.radians(deg / 2))
        return physical_size_mm / self.pixel_pitch_mm

    def load_jsonl(self, path):
        self.events = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): self.events.append(json.loads(line))
        if not self.events: return False
        
        first_event = self.events[0]
        targets = first_event.get("targets", [])
        raw_layout = first_event.get("layout", [])
        self.layout = {targets[i]: raw_layout[i] for i in range(min(len(targets), len(raw_layout)))}
        
        if "physical_setup" in first_event:
            self.screen_dist_mm = float(first_event["physical_setup"].get("distance_to_screen_mm", 600.0))
            self.pixel_pitch_mm = float(first_event["physical_setup"].get("monitor_pixel_pitch_mm", 0.27))

        for ev in reversed(self.events):
            if ev.get("elapsed_since_start_ms") is not None:
                self.max_time_ms = ev["elapsed_since_start_ms"]
                break
        self.is_json_loaded = True
        self.calibrate_mouse_coordinates()
        return True

    def load_asc(self, path, offset_x=0, offset_y=0):
        self.gaze_samples, self.calib_events, self.calib_gaze_matches = [], [], []
        sample_pattern = re.compile(r"^\s*(\d+)\s+([^\s]+)\s+([^\s]+)")
        msg_timer_pattern = re.compile(r"^MSG\s+(\d+)\s+TMT_EVENT:\s+timer_started")
        msg_calib_pattern = re.compile(r"^MSG\s+(\d+)\s+TMT_EVENT:\s+CALIBRATION_DOT_(\d+)_X:(\d+)_Y:(\d+)")

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        timer_syncs = [int(m.group(1)) for line in lines if (m := msg_timer_pattern.match(line.strip()))]
        target_idx = 0 if self.task_type == "A" else 1
        sync_time = timer_syncs[target_idx] if len(timer_syncs) > target_idx else (timer_syncs[-1] if timer_syncs else 0)

        calib_start = 0 if self.task_type == "A" else (timer_syncs[0] if len(timer_syncs) > 1 else 0)
        calib_end = sync_time

        for line in lines:
            s = line.strip()
            if calib_match := msg_calib_pattern.match(s):
                raw_ts = int(calib_match.group(1))
                if calib_start <= raw_ts <= calib_end:
                    self.calib_events.append((raw_ts - sync_time, int(calib_match.group(2)), int(calib_match.group(3)), int(calib_match.group(4))))
                continue

            if match := sample_pattern.match(s):
                ts_str, x_str, y_str = match.groups()
                if x_str == "." or y_str == ".": continue
                try: self.gaze_samples.append((int(ts_str) - sync_time, float(x_str), float(y_str)))
                except ValueError: pass

        self.min_time_ms = self.gaze_samples[0][0] if self.gaze_samples else 0

        if self.gaze_samples and self.calib_events:
            for ct, idx, cx, cy in self.calib_events:
                closest = min(self.gaze_samples, key=lambda g: abs(g[0] - ct))
                self.calib_gaze_matches.append({
                    "ts": ct, "idx": idx, "target_cx": cx, "target_cy": cy,
                    "raw_cx": closest[1] - offset_x, "raw_cy": closest[2] - offset_y
                })
        self.is_asc_loaded = True
        return True

    def calibrate_mouse_coordinates(self):
        clicks = [ev for ev in self.events if ev.get("event_type") == "correct_click" and "target" in ev and "x" in ev and "y" in ev]
        if len(clicks) < 2: return
        Ax, Bx, Ay, By = [], [], [], []
        for c in clicks:
            if str(c["target"]) in self.layout:
                pct_x, pct_y = self.layout[str(c["target"])]
                Ax.append([c["x"], 1]); Bx.append(pct_x); Ay.append([c["y"], 1]); By.append(pct_y)
        if len(Ax) >= 2:
            self.mouse_to_pct_m_x, self.mouse_to_pct_b_x = np.linalg.lstsq(Ax, Bx, rcond=None)[0][:2]
            self.mouse_to_pct_m_y, self.mouse_to_pct_b_y = np.linalg.lstsq(Ay, By, rcond=None)[0][:2]

    def apply_calibration(self, raw_cx, raw_cy, center_cx, center_cy):
        if self.use_auto_calib:
            x, y = raw_cx, raw_cy
            cx = self.calib_coef_x[0] + self.calib_coef_x[1]*x + self.calib_coef_x[2]*y + self.calib_coef_x[3]*(x**2) + self.calib_coef_x[4]*(y**2) + self.calib_coef_x[5]*x*y
            cy = self.calib_coef_y[0] + self.calib_coef_y[1]*x + self.calib_coef_y[2]*y + self.calib_coef_y[3]*(x**2) + self.calib_coef_y[4]*(y**2) + self.calib_coef_y[5]*x*y
            return cx, cy
        return ((raw_cx - center_cx) * self.gaze_scale_x) + center_cx + self.gaze_offset_x, ((raw_cy - center_cy) * self.gaze_scale_y) + center_cy + self.gaze_offset_y

class HeadlessCognitiveAnalyzer:
    def __init__(self, canvas_size):
        self.canvas_size = canvas_size

    def _get_node_canvas_pos(self, x, y):
        padding = self.canvas_size * 0.08
        active = self.canvas_size - (padding * 2.0)
        return int((x / 100.0) * active + padding), int((y / 100.0) * active + padding)

    def analyze_session(self, session, offset_x=0, offset_y=0):
        # Biologically defined radii
        fovea_radius_px = session.degrees_to_pixels(AOI_ANGLE_DEG) / 2.0
        peripheral_radius_px = session.degrees_to_pixels(PERIPHERAL_ANGLE_DEG) / 2.0

        clicks = []
        for ev in session.events:
            if ev.get("event_type") == "correct_click":
                t_id = ev["target"]
                cx, cy = self._get_node_canvas_pos(*session.layout[t_id])
                clicks.append({"id": t_id, "time": ev["elapsed_since_start_ms"], "cx": cx, "cy": cy})

        center_cx = self.canvas_size / 2.0
        calibrated_gaze = [(t, *session.apply_calibration(rx - offset_x, ry - offset_y, center_cx, center_cx)) for t, rx, ry in session.gaze_samples]

        task_memory, task_search, task_motor, task_saccades = [], [], [], []
        task_skips = 0
        task_capacities, task_noises_deg = [], []

        for i in range(1, len(clicks)):
            prev, curr = clicks[i-1], clicks[i]
            t_start, t_end = prev["time"], curr["time"]
            segment = [g for g in calibrated_gaze if t_start <= g[0] <= t_end]
            if not segment: continue

            # Working Memory Dwell inside Fovea
            t_leave = t_start
            for g in segment:
                if math.hypot(g[1] - prev["cx"], g[2] - prev["cy"]) > fovea_radius_px:
                    t_leave = g[0]
                    break
            task_memory.append(t_leave - t_start)

            t_fix_start, fix_timer, last_g_t, saccades = None, 0, t_leave, 0
            landing_gaze = None

            for g in segment:
                if g[0] < t_leave: continue
                dt = g[0] - last_g_t
                if dt > 0:
                    px_dist = math.hypot(g[1] - segment[segment.index(g)-1][1], g[2] - segment[segment.index(g)-1][2])
                    if (session.pixels_to_degrees(px_dist) / dt) > SACCADE_VELOCITY_DEG_MS:
                        saccades += 1
                last_g_t = g[0]

                if math.hypot(g[1] - curr["cx"], g[2] - curr["cy"]) <= fovea_radius_px:
                    if fix_timer == 0:
                        fix_start_t = g[0]
                        landing_gaze = g
                    fix_timer += dt
                    if fix_timer >= MIN_FIXATION_MS and t_fix_start is None:
                        t_fix_start = fix_start_t
                else:
                    if 0 < fix_timer < MIN_FIXATION_MS: task_skips += 1
                    fix_timer = 0

            # Spatial Memory Extraction
            if t_fix_start is not None and saccades <= 1 and landing_gaze is not None:
                n_px = math.hypot(landing_gaze[1] - curr["cx"], landing_gaze[2] - curr["cy"])
                task_noises_deg.append(session.pixels_to_degrees(n_px))
                
                t_seen = None
                for past_g in reversed([g for g in calibrated_gaze if g[0] < t_leave]):
                    if math.hypot(past_g[1] - curr["cx"], past_g[2] - curr["cy"]) < peripheral_radius_px:
                        t_seen = past_g[0]
                        break
                if t_seen is not None:
                    seen_step = next((k for k in range(i) if (0 if k == 0 else clicks[k-1]["time"]) <= t_seen <= clicks[k]["time"]), 0)
                    task_capacities.append(i - seen_step)

            if t_fix_start is None: t_fix_start = t_end
            task_search.append(t_fix_start - t_leave)
            task_motor.append(t_end - t_fix_start)
            task_saccades.append(saccades)

        return {
            "Memory (ms)": float(np.mean(task_memory)) if task_memory else 0.0,
            "Search (ms)": float(np.mean(task_search)) if task_search else 0.0,
            "Motor (ms)": float(np.mean(task_motor)) if task_motor else 0.0,
            "Total Skips": int(task_skips),
            "Saccades/Search": float(np.mean(task_saccades)) if task_saccades else 0.0,
            "Memory Capacity": int(np.mean(task_capacities)) if task_capacities else 1,
            "Memory Noise (deg)": float(np.mean(task_noises_deg)) if task_noises_deg else AOI_ANGLE_DEG
        }

class TrajectoryRewardPolicyTuner:
    """Embedded RL Tuner utilizing Cross-Entropy optimization to map human behavior."""
    def __init__(self, human_session, canvas_size):
        self.human_session = human_session
        self.task_type = human_session.task_type
        self.analyzer = HeadlessCognitiveAnalyzer(canvas_size=canvas_size)
        self.human_clicks = [ev for ev in human_session.events if ev.get("event_type") == "correct_click"]
        self.human_metrics = self.analyzer.analyze_session(human_session)

    def compute_behavioral_loss(self, policy_vector):
        params = {
            "Memory (ms)": policy_vector[0], "Search (ms)": policy_vector[1], "Motor (ms)": policy_vector[2],
            "Saccades/Search": policy_vector[3], "Memory Capacity": int(round(policy_vector[4])), "Memory Noise (deg)": policy_vector[5]
        }
        
        agent = CognitiveTMTAgent(
            task_type=self.task_type, participant_id="RL_Probe", custom_metrics=params,
            screen_dist_mm=self.human_session.screen_dist_mm, pixel_pitch_mm=self.human_session.pixel_pitch_mm
        )
        sim_json, sim_asc = agent.run_simulation()
        
        sim_session = TMTSession(self.task_type)
        sim_session.load_jsonl(sim_json)
        sim_session.load_asc(sim_asc)
        sim_metrics = self.analyzer.analyze_session(sim_session)
        
        sim_clicks = [ev for ev in sim_session.events if ev.get("event_type") == "correct_click"]
        time_diffs = [((h.get("elapsed_since_start_ms") or 0) - (s.get("elapsed_since_start_ms") or 0)) ** 2 for h, s in zip(self.human_clicks, sim_clicks)]
        timeline_loss = math.sqrt(np.mean(time_diffs)) / max(1.0, self.human_session.max_time_ms)
        
        metric_keys = ["Memory (ms)", "Search (ms)", "Motor (ms)", "Saccades/Search", "Memory Noise (deg)"]
        metric_errors = [((self.human_metrics.get(k, 0.0) - sim_metrics.get(k, 0.0)) / max(self.human_metrics.get(k, 1.0), 0.1)) ** 2 for k in metric_keys]
        
        Path(sim_json).unlink(missing_ok=True)
        Path(sim_asc).unlink(missing_ok=True)
        
        return (0.6 * timeline_loss) + (0.4 * np.mean(metric_errors)), sim_metrics

    def fit_parameters(self, max_episodes, population_size, callback):
        mu = np.array([
            max(20.0, self.human_metrics.get("Memory (ms)", 50.0)), max(300.0, self.human_metrics.get("Search (ms)", 1000.0)),
            max(150.0, self.human_metrics.get("Motor (ms)", 400.0)), max(2.0, self.human_metrics.get("Saccades/Search", 6.0)),
            float(max(1, self.human_metrics.get("Memory Capacity", 2))), max(0.5, self.human_metrics.get("Memory Noise (deg)", 1.5))
        ], dtype=float)
        
        sigma = np.array([20.0, 250.0, 80.0, 2.5, 0.8, 0.4], dtype=float)
        bounds_low = np.array([15.0, 250.0, 100.0, 1.0, 1.0, 0.3])
        bounds_high = np.array([250.0, 3500.0, 900.0, 25.0, 5.0, 4.0])
        
        best_loss, best_params = float('inf'), None
        for ep in range(max_episodes):
            cands = np.clip(np.random.normal(mu, sigma, size=(population_size, 6)), bounds_low, bounds_high)
            results = sorted([(self.compute_behavioral_loss(c)[0], c) for c in cands], key=lambda x: x[0])
            
            if results[0][0] < best_loss: best_loss, best_params = results[0][0], results[0][1]
            elite = np.array([x[1] for x in results[:max(2, population_size // 3)]])
            mu = 0.7 * mu + 0.3 * np.mean(elite, axis=0)
            sigma = 0.7 * sigma + 0.3 * np.std(elite, axis=0) + 1e-4
            
            if callback: callback(ep + 1, max_episodes, best_loss)
                
        return {
            "Memory (ms)": float(best_params[0]), "Search (ms)": float(best_params[1]), "Motor (ms)": float(best_params[2]),
            "Saccades/Search": float(best_params[3]), "Memory Capacity": int(round(best_params[4])),
            "Memory Noise (deg)": float(best_params[5]), "Fitting Loss": float(best_loss)
        }

class TMTReplayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TMT Biological Visualizer & RL Tuner")
        self.root.configure(bg="#2b2b2b")

        self.canvas_size = int(self.root.winfo_screenheight() * 0.85)
        self.node_radius = int(self.canvas_size * 0.025)

        self.sessions = {"A": TMTSession("A"), "B": TMTSession("B")}
        self.active_task, self.play_mode = "A", "A"
        self.current_time_ms, self.is_playing, self.playback_speed = 0, False, 1.0
        self.calib_active_task = None 

        self._build_ui()

    def _build_ui(self):
        ctrl = ttk.Frame(self.root, padding=10)
        ctrl.pack(fill=tk.X)

        for task in ["A", "B"]:
            row = ttk.Frame(ctrl)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"TMT-{task}:", font=("Segoe UI", 10, "bold"), width=8).pack(side=tk.LEFT)
            ttk.Button(row, text="Load JSONL", command=lambda t=task: self._load_jsonl_ui(t)).pack(side=tk.LEFT, padx=4)
            ttk.Button(row, text="Load ASC", command=lambda t=task: self._load_asc_ui(t)).pack(side=tk.LEFT, padx=4)
            btn = ttk.Button(row, text="Align Calibration", command=lambda t=task: self._open_calibration_window(t), state=tk.DISABLED)
            btn.pack(side=tk.LEFT, padx=4)
            if task == "A": self.calib_btn_a = btn
            else: self.calib_btn_b = btn
            lbl = ttk.Label(row, text="Waiting for files...", foreground="#4da6ff")
            lbl.pack(side=tk.LEFT, padx=10)
            if task == "A": self.status_lbl_a = lbl
            else: self.status_lbl_b = lbl

        ttk.Separator(ctrl, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        row_play = ttk.Frame(ctrl)
        row_play.pack(fill=tk.X, pady=2)
        ttk.Label(row_play, text="Playback:", font=("Segoe UI", 10, "bold"), width=8).pack(side=tk.LEFT)
        ttk.Button(row_play, text="Play A", command=lambda: self._start_playback("A")).pack(side=tk.LEFT, padx=4)
        ttk.Button(row_play, text="Play B", command=lambda: self._start_playback("B")).pack(side=tk.LEFT, padx=4)
        ttk.Button(row_play, text="Play Seq", command=lambda: self._start_playback("Seq")).pack(side=tk.LEFT, padx=4)
        ttk.Button(row_play, text="Stop", command=self._reset_playback).pack(side=tk.LEFT, padx=10)

        ttk.Label(row_play, text="Speed:").pack(side=tk.LEFT, padx=(5, 2))
        self.speed_var = tk.StringVar(value="1.0x")
        speed_menu = ttk.Combobox(row_play, textvariable=self.speed_var, values=["0.5x", "1.0x", "2.0x", "4.0x"], width=5)
        speed_menu.pack(side=tk.LEFT)
        speed_menu.bind("<<ComboboxSelected>>", lambda e: setattr(self, 'playback_speed', float(self.speed_var.get().replace("x", ""))))
        
        ttk.Button(row_play, text="Fit Parameters via RL/ML", command=self._run_rl_tuning).pack(side=tk.RIGHT, padx=6)
        ttk.Button(row_play, text="Algorithmic Analysis", command=self._run_algorithmic_analysis).pack(side=tk.RIGHT, padx=6)

        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg="#f5f5f5", highlightthickness=0)
        self.canvas.pack(pady=10)
        self.canvas.create_text(self.canvas_size/2, self.canvas_size/2, text="Load Data to Begin", font=("Segoe UI", 16, "bold"), fill="#aaaaaa")

    def _load_jsonl_ui(self, task):
        path = filedialog.askopenfilename(title=f"Select JSONL for TMT-{task}", filetypes=[("JSON Lines", "*.jsonl")])
        if path and self.sessions[task].load_jsonl(path):
            self.status_lbl_a.config(text=f"Loaded: {Path(path).name}") if task == "A" else self.status_lbl_b.config(text=f"Loaded: {Path(path).name}")

    def _load_asc_ui(self, task):
        path = filedialog.askopenfilename(title=f"Select ASC for TMT-{task}", filetypes=[("EyeLink ASC", "*.asc")])
        if path:
            off_x = (self.root.winfo_screenwidth() - self.canvas_size) / 2
            off_y = (self.root.winfo_screenheight() - self.canvas_size) / 2
            if self.sessions[task].load_asc(path, off_x, off_y):
                btn = self.calib_btn_a if task == "A" else self.calib_btn_b
                btn.config(state=tk.NORMAL)

    def _open_calibration_window(self, task_type):
        self.calib_active_task = task_type
        session = self.sessions[task_type]
        if not session.calib_gaze_matches: return

        calib_win = tk.Toplevel(self.root)
        calib_win.title(f"Gaze Calibration - TMT-{task_type}")
        calib_win.geometry(f"{self.canvas_size + 350}x{self.canvas_size + 50}")
        calib_win.configure(bg="#2b2b2b")

        ctrl_frame = ttk.Frame(calib_win, padding=10)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Button(ctrl_frame, text="✨ Auto Calibrate (Optimal Fit)", command=self._run_auto_calibration).pack(pady=(10, 30), fill=tk.X)
        
        ttk.Label(ctrl_frame, text="--- Manual Overrides ---").pack(pady=(0, 10))
        ttk.Label(ctrl_frame, text="X Offset (px):").pack()
        self.ox_scale = ttk.Scale(ctrl_frame, from_=-500, to=500, orient=tk.HORIZONTAL, command=self._manual_slider_update)
        self.ox_scale.set(session.gaze_offset_x)
        self.ox_scale.pack(fill=tk.X)
        
        ttk.Label(ctrl_frame, text="Y Offset (px):").pack(pady=(10, 0))
        self.oy_scale = ttk.Scale(ctrl_frame, from_=-500, to=500, orient=tk.HORIZONTAL, command=self._manual_slider_update)
        self.oy_scale.set(session.gaze_offset_y)
        self.oy_scale.pack(fill=tk.X)
        
        ttk.Label(ctrl_frame, text="X Scale:").pack(pady=(20, 0))
        self.sx_scale = ttk.Scale(ctrl_frame, from_=0.5, to=2.0, orient=tk.HORIZONTAL, command=self._manual_slider_update)
        self.sx_scale.set(session.gaze_scale_x)
        self.sx_scale.pack(fill=tk.X)
        
        ttk.Label(ctrl_frame, text="Y Scale:").pack(pady=(10, 0))
        self.sy_scale = ttk.Scale(ctrl_frame, from_=0.5, to=2.0, orient=tk.HORIZONTAL, command=self._manual_slider_update)
        self.sy_scale.set(session.gaze_scale_y)
        self.sy_scale.pack(fill=tk.X)
        
        self.calib_status_lbl = ttk.Label(ctrl_frame, text="Mode: Auto" if session.use_auto_calib else "Mode: Manual", foreground="#00cc00" if session.use_auto_calib else "#ffaa00")
        self.calib_status_lbl.pack(pady=20)
        ttk.Button(ctrl_frame, text="Reset to Default", command=self._reset_calib_sliders).pack(pady=10)
        ttk.Button(ctrl_frame, text="Apply & Close", command=calib_win.destroy).pack(pady=10)

        self.calib_canvas = tk.Canvas(calib_win, width=self.canvas_size, height=self.canvas_size, bg="#f5f5f5", highlightthickness=0)
        self.calib_canvas.pack(side=tk.RIGHT, padx=10, pady=10)
        self._redraw_calib_preview()

    def _run_auto_calibration(self):
        session = self.sessions[self.calib_active_task]
        A, Bx, By = [], [], []
        for match in session.calib_gaze_matches:
            x, y = match["raw_cx"], match["raw_cy"]
            A.append([1, x, y, x**2, y**2, x*y])
            Bx.append(match["target_cx"])
            By.append(match["target_cy"])
        A, Bx, By = np.array(A), np.array(Bx), np.array(By)
        session.calib_coef_x, _, _, _ = np.linalg.lstsq(A, Bx, rcond=None)
        session.calib_coef_y, _, _, _ = np.linalg.lstsq(A, By, rcond=None)
        session.use_auto_calib = True
        self.calib_status_lbl.config(text="Mode: Auto (Polynomial Fit)", foreground="#00cc00")
        self._redraw_calib_preview()

    def _manual_slider_update(self, event=None):
        session = self.sessions[self.calib_active_task]
        session.use_auto_calib = False
        if hasattr(self, 'calib_status_lbl') and self.calib_status_lbl.winfo_exists():
            self.calib_status_lbl.config(text="Mode: Manual (Linear)", foreground="#ffaa00")
        session.gaze_offset_x, session.gaze_offset_y = self.ox_scale.get(), self.oy_scale.get()
        session.gaze_scale_x, session.gaze_scale_y = self.sx_scale.get(), self.sy_scale.get()
        self._redraw_calib_preview()

    def _reset_calib_sliders(self):
        self.ox_scale.set(0); self.oy_scale.set(0); self.sx_scale.set(1.0); self.sy_scale.set(1.0)
        self._manual_slider_update()

    def _redraw_calib_preview(self):
        session = self.sessions[self.calib_active_task]
        self.calib_canvas.delete("all")
        self.calib_canvas.create_rectangle(0, 0, self.canvas_size, self.canvas_size, fill="#f8f8f8")
        center = self.canvas_size / 2
        for match in session.calib_gaze_matches:
            tcx, tcy = match["target_cx"], match["target_cy"]
            self.calib_canvas.create_oval(tcx-8, tcy-8, tcx+8, tcy+8, outline="#ff4d4d", width=2)
            trans_cx, trans_cy = session.apply_calibration(match["raw_cx"], match["raw_cy"], center, center)
            self.calib_canvas.create_line(tcx, tcy, trans_cx, trans_cy, fill="#aaaaaa", dash=(4, 4))
            self.calib_canvas.create_oval(trans_cx-5, trans_cy-5, trans_cx+5, trans_cy+5, fill="#3388ff")

    def _run_algorithmic_analysis(self):
        analyzer = HeadlessCognitiveAnalyzer(self.canvas_size)
        offset_x = (self.root.winfo_screenwidth() - self.canvas_size) / 2
        offset_y = (self.root.winfo_screenheight() - self.canvas_size) / 2
        results = {}
        for t in ["A", "B"]:
            s = self.sessions[t]
            if s.is_json_loaded and s.is_asc_loaded: results[t] = analyzer.analyze_session(s, offset_x, offset_y)
        self._show_results_dashboard(results, "Algorithmic Analysis (Analytical Slicing)")

    def _run_rl_tuning(self):
        target_task = self.active_task
        session = self.sessions[target_task]
        if not session.is_json_loaded or not session.is_asc_loaded:
            messagebox.showwarning("Data Required", f"Load both JSONL and ASC for TMT-{target_task} before running RL fitting.")
            return

        win = tk.Toplevel(self.root)
        win.title("Behavioral RL Optimization Progress")
        win.geometry("440x220")
        win.configure(bg="#2b2b2b")
        
        lbl_status = ttk.Label(win, text="Starting Imitation Policy Search...", font=("Segoe UI", 11, "bold"), foreground="white")
        lbl_status.pack(pady=15)
        prog = ttk.Progressbar(win, orient=tk.HORIZONTAL, length=320, mode='determinate')
        prog.pack(pady=10)
        lbl_loss = ttk.Label(win, text="Loss: Computing...", foreground="#4da6ff")
        lbl_loss.pack(pady=5)

        def worker():
            tuner = TrajectoryRewardPolicyTuner(session, self.canvas_size)
            def on_step(ep, max_ep, loss):
                prog['value'] = (ep / max_ep) * 100
                lbl_status.config(text=f"Optimization Episode {ep}/{max_ep}")
                lbl_loss.config(text=f"Current Behavioral Loss: {loss:.4f}")
                win.update_idletasks()

            fitted_metrics = tuner.fit_parameters(max_episodes=15, population_size=6, callback=on_step)
            win.destroy()
            self._show_results_dashboard({target_task: fitted_metrics}, f"RL Behavior-Fitted Model (TMT-{target_task})")

        threading.Thread(target=worker, daemon=True).start()

    def _show_results_dashboard(self, results, title):
        if not results: return
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("520x460")
        win.configure(bg="#2b2b2b")
        
        ttk.Label(win, text=title, font=("Segoe UI", 12, "bold"), foreground="white").pack(pady=10)
        for task, m in results.items():
            if not m: continue
            f = ttk.LabelFrame(win, text=f" TMT-{task} Model ", padding=10)
            f.pack(fill=tk.X, padx=15, pady=5)
            ttk.Label(f, text=f"Working Memory Speed: {m['Memory (ms)']:.1f} ms").pack(anchor=tk.W)
            ttk.Label(f, text=f"Visual Search Speed:   {m['Search (ms)']:.1f} ms").pack(anchor=tk.W)
            ttk.Label(f, text=f"Motor Execution Speed: {m['Motor (ms)']:.1f} ms").pack(anchor=tk.W)
            ttk.Label(f, text=f"Saccades Per Target:   {m['Saccades/Search']:.1f}").pack(anchor=tk.W)
            ttk.Label(f, text=f"Spatial Memory Buffer: {m['Memory Capacity']} targets").pack(anchor=tk.W)
            ttk.Label(f, text=f"Spatial Retinal Noise: {m['Memory Noise (deg)']:.2f}°").pack(anchor=tk.W)
            if "Fitting Loss" in m:
                ttk.Label(f, text=f"Scanpath Fidelity Loss: {m['Fitting Loss']:.4f}", foreground="#00cc66").pack(anchor=tk.W, pady=(4,0))

    def _start_playback(self, mode):
        self.play_mode = mode
        self.active_task = "A" if mode in ["A", "Seq"] else "B"
        if not self.sessions[self.active_task].is_json_loaded: return
        self.current_time_ms, self.is_playing = 0, True
        self._playback_loop()

    def _reset_playback(self):
        self.is_playing = False
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, self.canvas_size, self.canvas_size, fill="#f8f8f8")
        self.canvas.create_text(self.canvas_size/2, self.canvas_size/2, text="Playback Stopped", font=("Segoe UI", 16, "bold"), fill="#aaaaaa")

    def _playback_loop(self):
        if not self.is_playing: return
        s = self.sessions[self.active_task]
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, self.canvas_size, self.canvas_size, fill="#f8f8f8")

        count = next((ev.get("completed_count", 0) for ev in reversed(s.events) if ev.get("elapsed_since_start_ms", float('inf')) <= self.current_time_ms), 0)
        latest_mouse = next(((ev.get("x"), ev.get("y"), ev.get("event_type")) for ev in reversed(s.events) if ev.get("elapsed_since_start_ms", float('inf')) <= self.current_time_ms and "x" in ev), None)

        padding = self.canvas_size * 0.08
        active = self.canvas_size - (padding * 2.0)
        for idx, (target, coords) in enumerate(s.layout.items()):
            cx = int((coords[0] / 100.0) * active + padding)
            cy = int((coords[1] / 100.0) * active + padding)
            self.canvas.create_oval(cx - self.node_radius, cy - self.node_radius, cx + self.node_radius, cy + self.node_radius, fill="#90ee90" if idx < count else "#ffffff", outline="#333333", width=2)
            self.canvas.create_text(cx, cy, text=target, font=("Segoe UI", int(self.node_radius * 0.7), "bold"))

        off_x, off_y = (self.root.winfo_screenwidth() - self.canvas_size) / 2, (self.root.winfo_screenheight() - self.canvas_size) / 2
        center = self.canvas_size / 2.0

        if s.gaze_samples:
            w_start = max(s.min_time_ms, self.current_time_ms - 300)
            vis = [(x, y) for (t, x, y) in s.gaze_samples if w_start <= t <= self.current_time_ms]
            for gx, gy in vis:
                tcx, tcy = s.apply_calibration(gx - off_x, gy - off_y, center, center)
                self.canvas.create_oval(tcx - 3, tcy - 3, tcx + 3, tcy + 3, fill="#ff3366", outline="")

            if vis:
                t_pct_x, t_pct_y = s.layout[list(s.layout.keys())[min(count, len(s.layout) - 1)]]
                tx = int((t_pct_x / 100.0) * active + padding)
                ty = int((t_pct_y / 100.0) * active + padding)
                
                cgx, cgy = s.apply_calibration(vis[-1][0] - off_x, vis[-1][1] - off_y, center, center)
                vf_r = s.degrees_to_pixels(PERIPHERAL_ANGLE_DEG) / 2.0
                fov_r = s.degrees_to_pixels(AOI_ANGLE_DEG) / 2.0
                
                self.canvas.create_oval(cgx - vf_r, cgy - vf_r, cgx + vf_r, cgy + vf_r, outline="#3388ff", dash=(4,4), width=2)
                self.canvas.create_oval(cgx - fov_r, cgy - fov_r, cgx + fov_r, cgy + fov_r, outline="#ff3366", width=2)
                self.canvas.create_text(20, 20, anchor="nw", text=f"Phase: TMT-{self.active_task} | Time: {int(self.current_time_ms)}ms | Gaze Target Error: {s.pixels_to_degrees(math.hypot(cgx - tx, cgy - ty)):.1f}°", font=("Segoe UI", 12, "bold"), fill="#1b1b1b")

        if latest_mouse:
            mx = int(((latest_mouse[0] * s.mouse_to_pct_m_x + s.mouse_to_pct_b_x) / 100.0) * active + padding)
            my = int(((latest_mouse[1] * s.mouse_to_pct_m_y + s.mouse_to_pct_b_y) / 100.0) * active + padding)
            self.canvas.create_oval(mx - 4, my - 4, mx + 4, my + 4, fill="#00cc00" if latest_mouse[2] == "correct_click" else "#3388ff", outline="#000000")

        self.current_time_ms += 25 * self.playback_speed
        if s.max_time_ms > 0 and self.current_time_ms > s.max_time_ms:
            if self.play_mode == "Seq" and self.active_task == "A" and self.sessions["B"].is_json_loaded:
                self.active_task, self.current_time_ms = "B", 0
                self.root.after(25, self._playback_loop)
                return
            self.is_playing = False
            return
        self.root.after(25, self._playback_loop)

if __name__ == "__main__":
    root = tk.Tk()
    app = TMTReplayApp(root)
    root.mainloop()