import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk


def load_jsonl(path):
    events = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


class TMTReplayApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TMT Replay from Log - Lab Sync")
        
        # --- FULLSCREEN & THEME SYNC ---
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg="#2b2b2b")
        self.root.bind("<Escape>", lambda e: self.root.destroy())

        screen_height = self.root.winfo_screenheight()
        self.canvas_size = int(screen_height * 0.85)
        self.node_radius = int(self.canvas_size * 0.025)
        self.cursor_radius = max(4, int(self.canvas_size * 0.008))
        self.click_marker_radius = max(8, int(self.canvas_size * 0.015))

        self.log_path = None
        self.events = []
        self.event_index = 0
        self.running = False
        self.play_speed = 1.0
        self.cursor_item = None
        self.click_marker = None
        self.cursor_path_points = []
        self.cursor_path_item = None
        self.targets = []
        self.task_layout = {}
        self.completed_count = 0
        self.current_target = None

        self._build_ui()
        self._reset_canvas_state()

    def _build_ui(self):
        style = ttk.Style()
        style.configure("TFrame", background="#2b2b2b")
        style.configure("TLabel", background="#2b2b2b", foreground="white")

        toolbar = ttk.Frame(self.root, padding=10)
        toolbar.pack(fill=tk.X)

        ttk.Button(toolbar, text="Open Log", command=self._open_log).pack(side=tk.LEFT, padx=4)
        self.play_button = ttk.Button(toolbar, text="Play", command=self._toggle_play, state=tk.DISABLED)
        self.play_button.pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Stop", command=self._stop_play, state=tk.DISABLED).pack(side=tk.LEFT, padx=4)

        ttk.Label(toolbar, text="Speed:").pack(side=tk.LEFT, padx=(12, 4))
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_combo = ttk.Combobox(toolbar, textvariable=self.speed_var, values=[0.25, 0.5, 1.0, 2.0, 4.0], width=6)
        speed_combo.pack(side=tk.LEFT)
        speed_combo.bind("<<ComboboxSelected>>", self._update_speed)
        
        ttk.Button(toolbar, text="Exit (ESC)", command=self.root.destroy).pack(side=tk.RIGHT, padx=4)

        status_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        status_frame.pack(fill=tk.X)
        self.status_text = tk.StringVar(value="Open a TMT log file to begin replay")
        self.task_text = tk.StringVar(value="Task: --")
        self.position_text = tk.StringVar(value="Event 0 / 0")

        ttk.Label(status_frame, textvariable=self.task_text, font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)
        ttk.Label(status_frame, textvariable=self.position_text, font=("Segoe UI", 12)).pack(side=tk.LEFT, padx=16)
        ttk.Label(status_frame, textvariable=self.status_text, font=("Segoe UI", 12), foreground="#4da6ff").pack(side=tk.RIGHT)

        canvas_container = tk.Frame(self.root, bg="#2b2b2b")
        canvas_container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_container, width=self.canvas_size, height=self.canvas_size, bg="#f5f5f5", highlightthickness=0)
        self.canvas.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def _reset_canvas_state(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, self.canvas_size, self.canvas_size, fill="#f8f8f8", outline="#eeeeee")
        self.cursor_item = self.canvas.create_oval(0, 0, 0, 0, outline="", fill="")
        self.click_marker = None
        self.cursor_path_points = []
        self.cursor_path_item = None

    def _open_log(self):
        path = filedialog.askopenfilename(
            title="Open TMT replay log",
            filetypes=[("JSONL files", "*.jsonl"), ("All files", "*")],
            initialdir=str(Path.cwd()),
        )
        if not path:
            return
        self.log_path = Path(path)
        self._load_events()

    def _load_events(self):
        self.events = load_jsonl(self.log_path)
        if not self.events:
            self.status_text.set("Selected log file contains no events.")
            self.play_button.config(state=tk.DISABLED)
            return

        header = next((e for e in self.events if e.get("event_type") == "task_started"), self.events[0])
        self.targets = header.get("targets", [])
        layout = header.get("layout", [])
        self.task_layout = {}
        for idx, target in enumerate(self.targets):
            if idx < len(layout):
                self.task_layout[target] = tuple(layout[idx])

        self.event_index = 0
        self.completed_count = 0
        self.current_target = header.get("current_target")
        self.running = False
        self._reset_canvas_state()
        self._redraw_canvas()
        self.play_button.config(state=tk.NORMAL, text="Play")
        self.status_text.set(f"Loaded {self.log_path.name}. Ready to replay.")
        self.task_text.set(f"Task: {header.get('task_type', '--')}")
        self.position_text.set(f"Event 0 / {len(self.events)}")

    def _redraw_canvas(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, self.canvas_size, self.canvas_size, fill="#f8f8f8", outline="#eeeeee")

        padding = self.canvas_size * 0.08
        active_area = self.canvas_size - (padding * 2)

        for idx, target in enumerate(self.targets):
            x, y = self.task_layout.get(target, (50, 50))
            
            canvas_x = int((x / 100.0) * active_area + padding)
            canvas_y = int((y / 100.0) * active_area + padding)
            
            is_completed = idx < self.completed_count
            color = "#90ee90" if is_completed else "#ffffff"
            outline = "#2e8b57" if is_completed else "#333333"
            width = 2
            
            self.canvas.create_oval(
                canvas_x - self.node_radius,
                canvas_y - self.node_radius,
                canvas_x + self.node_radius,
                canvas_y + self.node_radius,
                fill=color,
                outline=outline,
                width=width,
            )
            self.canvas.create_text(canvas_x, canvas_y, text=target, fill="#111111", font=("Segoe UI", int(self.node_radius*0.7), "bold"))

        if self.cursor_item is not None:
            self.canvas.lift(self.cursor_item)
        if self.click_marker is not None:
            self.canvas.lift(self.click_marker)

    def _toggle_play(self):
        if not self.running:
            self.running = True
            self.play_button.config(text="Pause")
            self._schedule_next_event(0)
        else:
            self.running = False
            self.play_button.config(text="Play")
            self.status_text.set("Paused")

    def _stop_play(self):
        self.running = False
        self.event_index = 0
        self.play_button.config(text="Play")
        self._reset_canvas_state()
        self._redraw_canvas()
        self.position_text.set(f"Event 0 / {len(self.events)}")
        self.status_text.set("Stopped")

    def _update_speed(self, event=None):
        self.play_speed = max(0.1, float(self.speed_var.get()))

    def _schedule_next_event(self, delay_ms):
        if not self.running:
            return
        self.root.after(delay_ms, self._process_next_event)

    def _process_next_event(self):
        if not self.running or self.event_index >= len(self.events):
            self.running = False
            self.play_button.config(text="Play")
            self.status_text.set("Replay finished")
            return

        event = self.events[self.event_index]
        self._handle_event(event)
        self.event_index += 1
        self.position_text.set(f"Event {self.event_index} / {len(self.events)}")
        next_delay = 0
        if self.event_index < len(self.events):
            next_event = self.events[self.event_index]
            next_delay = int((next_event.get("event_delta_ms", 0) or 0) / self.play_speed)
        self._schedule_next_event(max(next_delay, 1))

    def _handle_event(self, event):
        event_type = event.get("event_type", "unknown")
        self.status_text.set(f"Replaying: {event_type}")

        if event_type == "task_started":
            self.completed_count = 0
            self.current_target = event.get("current_target")
            self._redraw_canvas()
        elif event_type == "mouse_move":
            self._move_cursor(event.get("x", 0), event.get("y", 0))
        elif event_type in {"correct_click", "incorrect_click", "miss_click", "click_after_completion"}:
            self._move_cursor(event.get("x", 0), event.get("y", 0))
            self._show_click(event.get("x", 0), event.get("y", 0), correct=(event_type == "correct_click"))
            self.completed_count = event.get("completed_count", self.completed_count)
            self.current_target = event.get("current_target")
            self._redraw_canvas()
        elif event_type == "task_completed":
            self.status_text.set("Task completed")
        elif event_type == "task_summary":
            self.status_text.set("Replay summary available")

    def _move_cursor(self, x, y):
        canvas_x = x
        canvas_y = y
        if self.cursor_item is None:
            self.cursor_item = self.canvas.create_oval(0, 0, 0, 0, fill="#ff3333", outline="")
        self.canvas.coords(
            self.cursor_item,
            canvas_x - self.cursor_radius,
            canvas_y - self.cursor_radius,
            canvas_x + self.cursor_radius,
            canvas_y + self.cursor_radius,
        )
        self.canvas.itemconfigure(self.cursor_item, fill="#ff3333")

        if self.cursor_path_points:
            last_x, last_y = self.cursor_path_points[-1]
            self.canvas.create_line(last_x, last_y, canvas_x, canvas_y, fill="#ff6666", width=2)
        self.cursor_path_points.append((canvas_x, canvas_y))

    def _show_click(self, x, y, correct=True):
        if self.click_marker is not None:
            self.canvas.delete(self.click_marker)
        outline = "#33cc33" if correct else "#ff3333"
        self.click_marker = self.canvas.create_oval(
            x - self.click_marker_radius,
            y - self.click_marker_radius,
            x + self.click_marker_radius,
            y + self.click_marker_radius,
            outline=outline,
            width=3,
        )
        self.root.after(300, self._clear_click_marker)

    def _clear_click_marker(self):
        if self.click_marker is not None:
            self.canvas.delete(self.click_marker)
            self.click_marker = None


def main():
    root = tk.Tk()
    app = TMTReplayApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()