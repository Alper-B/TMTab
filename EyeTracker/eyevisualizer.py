import json
import re
import tkinter as tk
from tkinter import filedialog, ttk
from pathlib import Path
import numpy as np
import math

# --- COGNITIVE ANALYSIS CONFIGURATION CONSTANTS ---
AOI_RADIUS_MULTIPLIER = 2.5       # Multiplier for node radius to define the visual Area of Interest (AOI)
MIN_FIXATION_MS = 100             # Minimum dwell time in milliseconds to count as a cognitive fixation
SACCADE_VELOCITY_THRESHOLD = 0.5  # Minimum velocity (pixels/ms) to be classified as a rapid saccadic eye movement

class TMTSession:
    def __init__(self, task_type):
        self.task_type = task_type
        self.events = []
        self.gaze_samples = []  
        self.calib_events = []  
        self.calib_gaze_matches = [] 
        self.layout = {}
        
        self.max_time_ms = 0
        self.min_time_ms = 0 
        self.is_json_loaded = False
        self.is_asc_loaded = False

        self.use_auto_calib = False
        self.calib_coef_x = [0, 1, 0, 0, 0, 0] 
        self.calib_coef_y = [0, 0, 1, 0, 0, 0] 
        
        self.gaze_offset_x = 0.0
        self.gaze_offset_y = 0.0
        self.gaze_scale_x = 1.0
        self.gaze_scale_y = 1.0

        # Mouse coordinate to percentage mapping
        self.mouse_to_pct_m_x = 1.0
        self.mouse_to_pct_b_x = 0.0
        self.mouse_to_pct_m_y = 1.0
        self.mouse_to_pct_b_y = 0.0

    def apply_calibration(self, raw_cx, raw_cy, center_cx, center_cy):
        if self.use_auto_calib:
            x, y = raw_cx, raw_cy
            cx = (self.calib_coef_x[0] + self.calib_coef_x[1]*x + self.calib_coef_x[2]*y + 
                  self.calib_coef_x[3]*(x**2) + self.calib_coef_x[4]*(y**2) + self.calib_coef_x[5]*x*y)
            cy = (self.calib_coef_y[0] + self.calib_coef_y[1]*x + self.calib_coef_y[2]*y + 
                  self.calib_coef_y[3]*(x**2) + self.calib_coef_y[4]*(y**2) + self.calib_coef_y[5]*x*y)
            return cx, cy
        else:
            cx = ((raw_cx - center_cx) * self.gaze_scale_x) + center_cx + self.gaze_offset_x
            cy = ((raw_cy - center_cy) * self.gaze_scale_y) + center_cy + self.gaze_offset_y
            return cx, cy

class TMTReplayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TMT Dual-Session Visualizer & Analyzer")
        self.root.configure(bg="#2b2b2b")

        screen_height = self.root.winfo_screenheight()
        self.canvas_size = int(screen_height * 0.85)
        self.node_radius = int(self.canvas_size * 0.025)

        self.sessions = {
            "A": TMTSession("A"),
            "B": TMTSession("B")
        }
        
        self.active_task = "A" 
        self.play_mode = "A" 
        self.current_time_ms = 0
        self.is_playing = False
        self.playback_speed = 1.0
        self.calib_active_task = None 

        self._build_ui()

    def _build_ui(self):
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill=tk.X)

        # --- TMT-A Row ---
        row_a = ttk.Frame(control_frame)
        row_a.pack(fill=tk.X, pady=2)
        ttk.Label(row_a, text="TMT-A:", font=("Segoe UI", 10, "bold"), width=8).pack(side=tk.LEFT)
        ttk.Button(row_a, text="Load JSONL", command=lambda: self._load_jsonl("A")).pack(side=tk.LEFT, padx=4)
        ttk.Button(row_a, text="Load ASC", command=lambda: self._load_asc("A")).pack(side=tk.LEFT, padx=4)
        self.calib_btn_a = ttk.Button(row_a, text="Align Calibration", command=lambda: self._open_calibration_window("A"), state=tk.DISABLED)
        self.calib_btn_a.pack(side=tk.LEFT, padx=4)
        self.status_lbl_a = ttk.Label(row_a, text="Waiting for files...", foreground="#4da6ff")
        self.status_lbl_a.pack(side=tk.LEFT, padx=10)

        # --- TMT-B Row ---
        row_b = ttk.Frame(control_frame)
        row_b.pack(fill=tk.X, pady=2)
        ttk.Label(row_b, text="TMT-B:", font=("Segoe UI", 10, "bold"), width=8).pack(side=tk.LEFT)
        ttk.Button(row_b, text="Load JSONL", command=lambda: self._load_jsonl("B")).pack(side=tk.LEFT, padx=4)
        ttk.Button(row_b, text="Load ASC", command=lambda: self._load_asc("B")).pack(side=tk.LEFT, padx=4)
        self.calib_btn_b = ttk.Button(row_b, text="Align Calibration", command=lambda: self._open_calibration_window("B"), state=tk.DISABLED)
        self.calib_btn_b.pack(side=tk.LEFT, padx=4)
        self.status_lbl_b = ttk.Label(row_b, text="Waiting for files...", foreground="#4da6ff")
        self.status_lbl_b.pack(side=tk.LEFT, padx=10)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        # --- Playback & Analysis Row ---
        row_play = ttk.Frame(control_frame)
        row_play.pack(fill=tk.X, pady=2)
        
        ttk.Label(row_play, text="Playback:", font=("Segoe UI", 10, "bold"), width=8).pack(side=tk.LEFT)
        self.play_a_btn = ttk.Button(row_play, text="Play A", command=lambda: self._start_playback("A"))
        self.play_a_btn.pack(side=tk.LEFT, padx=4)
        self.play_b_btn = ttk.Button(row_play, text="Play B", command=lambda: self._start_playback("B"))
        self.play_b_btn.pack(side=tk.LEFT, padx=4)
        self.play_seq_btn = ttk.Button(row_play, text="Play Sequence", command=lambda: self._start_playback("Seq"))
        self.play_seq_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(row_play, text="Stop / Reset", command=self._reset_playback).pack(side=tk.LEFT, padx=10)

        ttk.Label(row_play, text="Speed:").pack(side=tk.LEFT, padx=(5, 2))
        self.speed_var = tk.StringVar(value="1.0x")
        speed_menu = ttk.Combobox(row_play, textvariable=self.speed_var, values=["0.5x", "1.0x", "2.0x", "4.0x"], width=5)
        speed_menu.pack(side=tk.LEFT)
        speed_menu.bind("<<ComboboxSelected>>", self._change_speed)
        
        # --- THE MAGIC ANALYSIS BUTTON ---
        self.analyze_btn = ttk.Button(row_play, text="Run Cognitive Analysis", command=self._run_analysis)
        self.analyze_btn.pack(side=tk.RIGHT, padx=10)

        canvas_container = tk.Frame(self.root, bg="#2b2b2b")
        canvas_container.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_container, width=self.canvas_size, height=self.canvas_size, bg="#f5f5f5", highlightthickness=0)
        self.canvas.pack(pady=10)
        self.canvas.create_text(self.canvas_size/2, self.canvas_size/2, text="Load Data to Begin", font=("Segoe UI", 16, "bold"), fill="#aaaaaa")

    def _calibrate_mouse_coordinates(self, session):
        # Auto-maps recorded mouse space to percentage space using correct clicks
        clicks = [ev for ev in session.events if ev.get("event_type") == "correct_click" and "target" in ev and "x" in ev and "y" in ev]
        if len(clicks) < 2:
            return

        Ax, Bx = [], []
        Ay, By = [], []
        for c in clicks:
            t_id = str(c["target"])
            if t_id in session.layout:
                pct_x, pct_y = session.layout[t_id]
                Ax.append([c["x"], 1])
                Bx.append(pct_x)
                Ay.append([c["y"], 1])
                By.append(pct_y)
        
        if len(Ax) >= 2:
            coef_x, _, _, _ = np.linalg.lstsq(Ax, Bx, rcond=None)
            session.mouse_to_pct_m_x = coef_x[0]
            session.mouse_to_pct_b_x = coef_x[1]
            
            coef_y, _, _, _ = np.linalg.lstsq(Ay, By, rcond=None)
            session.mouse_to_pct_m_y = coef_y[0]
            session.mouse_to_pct_b_y = coef_y[1]

    def _load_jsonl(self, task_type):
        path = filedialog.askopenfilename(title=f"Select JSONL for TMT-{task_type}", filetypes=[("JSON Lines", "*.jsonl"), ("JSON files", "*.json")])
        if not path: return
        session = self.sessions[task_type]
        session.events = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip(): session.events.append(json.loads(line))
        if not session.events: return
        first_event = session.events[0]
        targets = first_event.get("targets", [])
        raw_layout = first_event.get("layout", [])
        session.layout = {targets[i]: raw_layout[i] for i in range(min(len(targets), len(raw_layout)))}
        for ev in reversed(session.events):
            if ev.get("elapsed_since_start_ms") is not None:
                session.max_time_ms = ev["elapsed_since_start_ms"]
                break
        session.is_json_loaded = True
        self._calibrate_mouse_coordinates(session)
        self._update_status_label(task_type)

    def _load_asc(self, task_type):
        path = filedialog.askopenfilename(title=f"Select ASC for TMT-{task_type}", filetypes=[("EyeLink ASC files", "*.asc"), ("Text files", "*.txt")])
        if not path: return
        session = self.sessions[task_type]
        session.gaze_samples = []
        session.calib_events = []
        session.calib_gaze_matches = []
        
        sample_pattern = re.compile(r"^\s*(\d+)\s+([^\s]+)\s+([^\s]+)")
        msg_timer_pattern = re.compile(r"^MSG\s+(\d+)\s+TMT_EVENT:\s+timer_started")
        msg_calib_pattern = re.compile(r"^MSG\s+(\d+)\s+TMT_EVENT:\s+CALIBRATION_DOT_(\d+)_X:(\d+)_Y:(\d+)")

        sync_time, first_ts = None, None
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if timer_match := msg_timer_pattern.match(s):
                    sync_time = int(timer_match.group(1))
                    break
                if not first_ts and (match := sample_pattern.match(s)):
                    first_ts = int(match.group(1))

        if sync_time is None: sync_time = first_ts if first_ts else 0

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
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

        session.min_time_ms = session.gaze_samples[0][0] if session.gaze_samples else 0

        if session.gaze_samples and session.calib_events:
            offset_x, offset_y = (self.root.winfo_screenwidth() - self.canvas_size) / 2, (self.root.winfo_screenheight() - self.canvas_size) / 2
            for ct, idx, cx, cy in session.calib_events:
                closest_gaze = min(session.gaze_samples, key=lambda g: abs(g[0] - ct))
                session.calib_gaze_matches.append({
                    "ts": ct, "idx": idx, "target_cx": cx, "target_cy": cy,
                    "raw_cx": closest_gaze[1] - offset_x, "raw_cy": closest_gaze[2] - offset_y
                })
            (self.calib_btn_a if task_type == "A" else self.calib_btn_b).config(state=tk.NORMAL) 

        session.is_asc_loaded = True
        self._update_status_label(task_type)

    def _update_status_label(self, task_type):
        session = self.sessions[task_type]
        lbl = self.status_lbl_a if task_type == "A" else self.status_lbl_b
        parts = []
        if session.is_json_loaded: parts.append(f"Logs: {len(session.events)} ev")
        if session.is_asc_loaded: parts.append(f"Gaze: {len(session.gaze_samples)} pts")
        lbl.config(text=" | ".join(parts) if parts else "Waiting for files...")

    def _open_calibration_window(self, task_type):
        self.calib_active_task = task_type
        session = self.sessions[task_type]
        if not session.calib_gaze_matches: return

        calib_win = tk.Toplevel(self.root)
        calib_win.title(f"Gaze Calibration Alignment - TMT-{task_type}")
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
        self.calib_status_lbl.config(text="Mode: Manual (Linear)", foreground="#ffaa00")
        session.gaze_offset_x, session.gaze_offset_y = self.ox_scale.get(), self.oy_scale.get()
        session.gaze_scale_x, session.gaze_scale_y = self.sx_scale.get(), self.sy_scale.get()
        self._redraw_calib_preview()

    def _reset_calib_sliders(self):
        self.ox_scale.set(0); self.oy_scale.set(0); self.sx_scale.set(1.0); self.sy_scale.set(1.0)
        self._manual_slider_update()

    def _redraw_calib_preview(self, event=None):
        session = self.sessions[self.calib_active_task]
        self.calib_canvas.delete("all")
        self.calib_canvas.create_rectangle(0, 0, self.canvas_size, self.canvas_size, fill="#f8f8f8", outline="#eeeeee")
        center = self.canvas_size / 2
        for match in session.calib_gaze_matches:
            tcx, tcy = match["target_cx"], match["target_cy"]
            self.calib_canvas.create_oval(tcx-8, tcy-8, tcx+8, tcy+8, outline="#ff4d4d", width=2)
            trans_cx, trans_cy = session.apply_calibration(match["raw_cx"], match["raw_cy"], center, center)
            self.calib_canvas.create_line(tcx, tcy, trans_cx, trans_cy, fill="#aaaaaa", dash=(4, 4))
            self.calib_canvas.create_oval(trans_cx-5, trans_cy-5, trans_cx+5, trans_cy+5, fill="#3388ff")

    def _get_node_canvas_pos(self, x, y):
        padding = self.canvas_size * 0.08
        active_area = self.canvas_size - (padding * 2)
        return int((x / 100.0) * active_area + padding), int((y / 100.0) * active_area + padding)

    def _run_analysis(self):
        """Processes the timelines to extract cognitive simulation metrics."""
        results = {}
        AOI_RADIUS = self.node_radius * AOI_RADIUS_MULTIPLIER 
        
        center_cx = self.canvas_size / 2
        offset_x = (self.root.winfo_screenwidth() - self.canvas_size) / 2
        offset_y = (self.root.winfo_screenheight() - self.canvas_size) / 2

        for task_name in ["A", "B"]:
            session = self.sessions[task_name]
            if not session.is_json_loaded or not session.is_asc_loaded:
                continue

            clicks = []
            for ev in session.events:
                if ev.get("event_type") == "correct_click":
                    t_id = ev["target"]
                    cx, cy = self._get_node_canvas_pos(*session.layout[t_id])
                    clicks.append({"id": t_id, "time": ev["elapsed_since_start_ms"], "cx": cx, "cy": cy})

            calibrated_gaze = []
            for t, raw_x, raw_y in session.gaze_samples:
                cx, cy = session.apply_calibration(raw_x - offset_x, raw_y - offset_y, center_cx, center_cx)
                calibrated_gaze.append((t, cx, cy))

            task_memory_times = []
            task_search_times = []
            task_motor_times = []
            task_skips = 0
            task_search_saccades = []

            for i in range(1, len(clicks)):
                prev_click = clicks[i-1]
                curr_click = clicks[i]
                
                t_start = prev_click["time"]
                t_end = curr_click["time"]
                
                segment = [g for g in calibrated_gaze if t_start <= g[0] <= t_end]
                if not segment: continue

                t_leave = t_start
                for g in segment:
                    dist = math.hypot(g[1] - prev_click["cx"], g[2] - prev_click["cy"])
                    if dist > AOI_RADIUS:
                        t_leave = g[0]
                        break
                task_memory_times.append(t_leave - t_start)

                t_fix_start = None
                fixation_timer = 0
                last_g_time = t_leave
                
                saccade_count = 0
                
                for g in segment:
                    if g[0] < t_leave: continue
                    
                    time_delta = g[0] - last_g_time
                    if time_delta > 0:
                        velocity = math.hypot(g[1] - segment[segment.index(g)-1][1], g[2] - segment[segment.index(g)-1][2]) / time_delta
                        if velocity > SACCADE_VELOCITY_THRESHOLD: 
                            saccade_count += 1
                    last_g_time = g[0]

                    dist = math.hypot(g[1] - curr_click["cx"], g[2] - curr_click["cy"])
                    if dist <= AOI_RADIUS:
                        if fixation_timer == 0:
                            fixation_start_t = g[0]
                        fixation_timer += time_delta
                        
                        if fixation_timer >= MIN_FIXATION_MS and t_fix_start is None:
                            t_fix_start = fixation_start_t
                    else:
                        if 0 < fixation_timer < MIN_FIXATION_MS:
                            task_skips += 1
                        fixation_timer = 0

                if t_fix_start is None: 
                    t_fix_start = t_end

                task_search_times.append(t_fix_start - t_leave)
                task_motor_times.append(t_end - t_fix_start)
                task_search_saccades.append(saccade_count)

            results[task_name] = {
                "Memory (ms)": np.mean(task_memory_times) if task_memory_times else 0,
                "Search (ms)": np.mean(task_search_times) if task_search_times else 0,
                "Motor (ms)": np.mean(task_motor_times) if task_motor_times else 0,
                "Total Skips": task_skips,
                "Saccades/Search": np.mean(task_search_saccades) if task_search_saccades else 0
            }

        self._show_analysis_dashboard(results)

    def _show_analysis_dashboard(self, results):
        if not results: return
        win = tk.Toplevel(self.root)
        win.title("Cognitive Simulation Metrics Dashboard")
        win.geometry("500x400")
        win.configure(bg="#2b2b2b")
        
        ttk.Label(win, text="Data Extracted from Timeline Slicer", font=("Segoe UI", 14, "bold")).pack(pady=10)

        for task, metrics in results.items():
            if not metrics: continue
            frame = ttk.LabelFrame(win, text=f" TMT-{task} Metrics ", padding=10)
            frame.pack(fill=tk.X, padx=20, pady=5)
            
            ttk.Label(frame, text=f"Working Memory (memory_speed): {int(metrics['Memory (ms)'])} ms").pack(anchor=tk.W)
            ttk.Label(frame, text=f"Visual Search (search_speed): {int(metrics['Search (ms)'])} ms").pack(anchor=tk.W)
            ttk.Label(frame, text=f"Motor Execution (motor_speed): {int(metrics['Motor (ms)'])} ms").pack(anchor=tk.W)
            ttk.Label(frame, text=f"Visual Skip Probability: {metrics['Total Skips']} occurrences").pack(anchor=tk.W)
            ttk.Label(frame, text=f"Avg Saccades per Search: {metrics['Saccades/Search']:.1f}").pack(anchor=tk.W)

        if "A" in results and "B" in results and results["A"] and results["B"]:
            shift_delta = (results["B"]["Search (ms)"] + results["B"]["Memory (ms)"]) - (results["A"]["Search (ms)"] + results["A"]["Memory (ms)"])
            
            shift_frame = ttk.LabelFrame(win, text=" Set Shifting Analysis (TMT-B vs TMT-A) ", padding=10)
            shift_frame.pack(fill=tk.X, padx=20, pady=10)
            ttk.Label(shift_frame, text=f"Set Shifting Time (shift_speed): {int(max(0, shift_delta))} ms", foreground="#ff4d4d", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W)

    def _draw_task_layout(self, session, completed_count=0):
        for idx, (target, coords) in enumerate(session.layout.items()):
            canvas_x, canvas_y = self._get_node_canvas_pos(coords[0], coords[1])
            is_completed = idx < completed_count
            color = "#90ee90" if is_completed else "#ffffff"
            outline = "#2e8b57" if is_completed else "#333333"
            self.canvas.create_oval(canvas_x - self.node_radius, canvas_y - self.node_radius, canvas_x + self.node_radius, canvas_y + self.node_radius, fill=color, outline=outline, width=2)
            self.canvas.create_text(canvas_x, canvas_y, text=target, fill="#111111", font=("Segoe UI", int(self.node_radius * 0.7), "bold"))

    def _start_playback(self, mode):
        self.play_mode = mode
        if mode == "Seq":
            if self.sessions["A"].is_json_loaded: self.active_task = "A"
            elif self.sessions["B"].is_json_loaded: self.active_task = "B"
            else: return 
        else:
            self.active_task = mode
            if not self.sessions[mode].is_json_loaded: return
                
        self.current_time_ms = 0
        self.is_playing = True
        self._playback_loop()

    def _reset_playback(self):
        self.is_playing = False
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, self.canvas_size, self.canvas_size, fill="#f8f8f8", outline="#eeeeee")
        self.canvas.create_text(self.canvas_size/2, self.canvas_size/2, text="Playback Stopped", font=("Segoe UI", 16, "bold"), fill="#aaaaaa")

    def _change_speed(self, event=None):
        self.playback_speed = float(self.speed_var.get().replace("x", ""))

    def _playback_loop(self):
        if not self.is_playing: return
        session = self.sessions[self.active_task]
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, self.canvas_size, self.canvas_size, fill="#f8f8f8", outline="#eeeeee")

        latest_mouse = None
        completed_count = 0
        
        for ev in session.events:
            ev_t = ev.get("elapsed_since_start_ms")
            if ev_t is not None and ev_t <= self.current_time_ms:
                completed_count = ev.get("completed_count", completed_count)
                if "x" in ev and "y" in ev:
                    latest_mouse = (ev["x"], ev["y"], ev["event_type"])
            elif ev_t is not None and ev_t > self.current_time_ms: break

        self._draw_task_layout(session, completed_count)
        self.canvas.create_text(20, 20, anchor="nw", text=f"Phase: TMT-{self.active_task} Task | Time: {int(self.current_time_ms)}ms", font=("Segoe UI", 12, "bold"), fill="#1b1b1b")

        if session.gaze_samples:
            window_start = max(session.min_time_ms, self.current_time_ms - 300)
            visible_gaze = [(x, y) for (t, x, y) in session.gaze_samples if window_start <= t <= self.current_time_ms]
            center_cx, offset_x, offset_y = self.canvas_size / 2, (self.root.winfo_screenwidth() - self.canvas_size) / 2, (self.root.winfo_screenheight() - self.canvas_size) / 2

            for gx, gy in visible_gaze:
                transformed_cx, transformed_cy = session.apply_calibration(gx - offset_x, gy - offset_y, center_cx, center_cx)
                self.canvas.create_oval(transformed_cx - 3, transformed_cy - 3, transformed_cx + 3, transformed_cy + 3, fill="#ff3366", outline="")

        if latest_mouse and self.current_time_ms >= 0:
            raw_mx, raw_my, ev_type = latest_mouse
            
            # Map raw coordinates to internal percentages, then back to the current canvas space
            pct_x = raw_mx * session.mouse_to_pct_m_x + session.mouse_to_pct_b_x
            pct_y = raw_my * session.mouse_to_pct_m_y + session.mouse_to_pct_b_y
            mx, my = self._get_node_canvas_pos(pct_x, pct_y)
            
            self.canvas.create_oval(mx - 4, my - 4, mx + 4, my + 4, fill="#00cc00" if ev_type == "correct_click" else "#3388ff", outline="#000000")

        step_interval = 25  
        self.current_time_ms += step_interval * self.playback_speed

        if session.max_time_ms > 0 and self.current_time_ms > session.max_time_ms:
            if self.play_mode == "Seq" and self.active_task == "A" and self.sessions["B"].is_json_loaded:
                self.active_task, self.current_time_ms = "B", 0
                self.root.after(step_interval, self._playback_loop)
                return
            self.is_playing = False
            self.canvas.create_text(self.canvas_size/2, 50, text="Playback Complete", font=("Segoe UI", 16, "bold"), fill="#2e8b57")
            return

        self.root.after(step_interval, self._playback_loop)

def main():
    root = tk.Tk()
    app = TMTReplayApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()