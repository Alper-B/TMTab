import json
import re
import tkinter as tk
from tkinter import filedialog, ttk
from pathlib import Path

class TMTReplayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TMT Experiment & Eye Gaze Visualizer")
        self.root.configure(bg="#2b2b2b")

        screen_height = self.root.winfo_screenheight()
        self.canvas_size = int(screen_height * 0.85)
        self.node_radius = int(self.canvas_size * 0.025)

        self.events = []
        self.gaze_samples = []
        self.layout = {}
        self.task_type = "A"
        
        self.current_time_ms = 0
        self.max_time_ms = 0
        self.is_playing = False
        self.playback_speed = 1.0

        self._build_ui()

    def _build_ui(self):
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill=tk.X)

        ttk.Button(control_frame, text="1. Load .JSONL Log", command=self._load_jsonl).pack(side=tk.LEFT, padx=4)
        ttk.Button(control_frame, text="2. Load .ASC Gaze", command=self._load_asc).pack(side=tk.LEFT, padx=4)
        
        self.play_btn = ttk.Button(control_frame, text="Play", command=self._toggle_play, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=4)
        
        ttk.Button(control_frame, text="Reset", command=self._reset_playback).pack(side=tk.LEFT, padx=4)

        ttk.Label(control_frame, text="Speed:").pack(side=tk.LEFT, padx=(10, 2))
        self.speed_var = tk.StringVar(value="1.0x")
        speed_menu = ttk.Combobox(control_frame, textvariable=self.speed_var, values=["0.5x", "1.0x", "2.0x", "4.0x"], width=5)
        speed_menu.pack(side=tk.LEFT)
        speed_menu.bind("<<ComboboxSelected>>", self._change_speed)

        self.status_lbl = ttk.Label(control_frame, text="Load JSONL to begin", foreground="#4da6ff")
        self.status_lbl.pack(side=tk.RIGHT, padx=10)

        canvas_container = tk.Frame(self.root, bg="#2b2b2b")
        canvas_container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_container, width=self.canvas_size, height=self.canvas_size, bg="#f5f5f5", highlightthickness=0)
        self.canvas.pack(pady=10)

    def _load_jsonl(self):
        path = filedialog.askopenfilename(filetypes=[("JSON Lines", "*.jsonl"), ("JSON files", "*.json")])
        if not path:
            return

        self.events = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.events.append(json.loads(line))

        if not self.events:
            return

        first_event = self.events[0]
        self.task_type = first_event.get("task_type", "A")
        targets = first_event.get("targets", [])
        raw_layout = first_event.get("layout", [])
        self.layout = {targets[i]: raw_layout[i] for i in range(min(len(targets), len(raw_layout)))}

        for ev in reversed(self.events):
            if ev.get("elapsed_since_start_ms") is not None:
                self.max_time_ms = ev["elapsed_since_start_ms"]
                break

        self.play_btn.config(state=tk.NORMAL)
        self.status_lbl.config(text=f"Loaded JSONL: {len(self.events)} events ({int(self.max_time_ms/1000)}s)")
        self._reset_playback()

    def _load_asc(self):
        path = filedialog.askopenfilename(filetypes=[("EyeLink ASC files", "*.asc"), ("Text files", "*.txt")])
        if not path:
            return

        self.gaze_samples = []
        start_sync_time = None
        raw_samples = []

        msg_pattern = re.compile(r"^MSG\s+(\d+)\s+(.+)")

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue

                if line_str.startswith("MSG"):
                    msg_match = msg_pattern.match(line_str)
                    if msg_match and "timer_started" in msg_match.group(2) and start_sync_time is None:
                        start_sync_time = int(msg_match.group(1))
                    continue

                parts = line_str.split()
                if len(parts) >= 3 and parts[0].isdigit():
                    ts_str, x_str, y_str = parts[0], parts[1], parts[2]
                    
                    if x_str == "." or y_str == ".":
                        continue
                    
                    try:
                        ts = int(ts_str)
                        gx = float(x_str)
                        gy = float(y_str)
                        raw_samples.append((ts, gx, gy))
                    except ValueError:
                        continue

        if not raw_samples:
            self.status_lbl.config(text="No valid gaze samples found in file.")
            return

        base_time = start_sync_time if start_sync_time is not None else raw_samples[0][0]
        self.gaze_samples = [
            (ts - base_time, gx, gy)
            for (ts, gx, gy) in raw_samples
            if (ts - base_time) >= 0
        ]

        self.status_lbl.config(text=f"Loaded {len(self.gaze_samples)} synced gaze points")

    def _get_node_canvas_pos(self, x, y):
        padding = self.canvas_size * 0.08
        active_area = self.canvas_size - (padding * 2)
        canvas_x = int((x / 100.0) * active_area + padding)
        canvas_y = int((y / 100.0) * active_area + padding)
        return canvas_x, canvas_y

    def _draw_layout(self, completed_count=0):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, self.canvas_size, self.canvas_size, fill="#f8f8f8", outline="#eeeeee")

        for idx, (target, coords) in enumerate(self.layout.items()):
            canvas_x, canvas_y = self._get_node_canvas_pos(coords[0], coords[1])
            is_completed = idx < completed_count

            color = "#90ee90" if is_completed else "#ffffff"
            outline = "#2e8b57" if is_completed else "#333333"

            self.canvas.create_oval(
                canvas_x - self.node_radius, canvas_y - self.node_radius,
                canvas_x + self.node_radius, canvas_y + self.node_radius,
                fill=color, outline=outline, width=2
            )
            self.canvas.create_text(
                canvas_x, canvas_y, text=target, fill="#111111",
                font=("Segoe UI", int(self.node_radius * 0.7), "bold")
            )

    def _toggle_play(self):
        self.is_playing = not self.is_playing
        self.play_btn.config(text="Pause" if self.is_playing else "Play")
        if self.is_playing:
            self._playback_loop()

    def _reset_playback(self):
        self.is_playing = False
        self.current_time_ms = 0
        self.play_btn.config(text="Play")
        self._draw_layout(0)

    def _change_speed(self, event=None):
        self.playback_speed = float(self.speed_var.get().replace("x", ""))

    def _playback_loop(self):
        if not self.is_playing:
            return

        completed_count = 0
        latest_mouse = None
        for ev in self.events:
            ev_t = ev.get("elapsed_since_start_ms")
            if ev_t is not None and ev_t <= self.current_time_ms:
                completed_count = ev.get("completed_count", completed_count)
                if "x" in ev and "y" in ev:
                    latest_mouse = (ev["x"], ev["y"], ev["event_type"])
            elif ev_t is not None and ev_t > self.current_time_ms:
                break

        self._draw_layout(completed_count)

        if self.gaze_samples:
            window_start = max(0, self.current_time_ms - 300)
            visible_gaze = [
                (x, y) for (t, x, y) in self.gaze_samples
                if window_start <= t <= self.current_time_ms
            ]

            screen_w = self.root.winfo_screenwidth()
            screen_h = self.root.winfo_screenheight()
            offset_x = (screen_w - self.canvas_size) / 2
            offset_y = (screen_h - self.canvas_size) / 2

            for gx, gy in visible_gaze:
                cx = gx - offset_x
                cy = gy - offset_y
                self.canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill="#ff3366", outline="")

        if latest_mouse:
            mx, my, ev_type = latest_mouse
            color = "#00cc00" if ev_type == "correct_click" else "#3388ff"
            r = 7 if "click" in ev_type else 4
            self.canvas.create_oval(mx - r, my - r, mx + r, my + r, fill=color, outline="#000000")

        step_interval = 25
        self.current_time_ms += step_interval * self.playback_speed

        if self.max_time_ms > 0 and self.current_time_ms > self.max_time_ms:
            self.is_playing = False
            self.play_btn.config(text="Play")
            return

        self.root.after(step_interval, self._playback_loop)

def main():
    root = tk.Tk()
    app = TMTReplayApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()