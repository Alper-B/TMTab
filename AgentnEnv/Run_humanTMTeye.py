import json
import sys
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk

# --- EYELINK PSYCHOPY IMPLEMENTATION ---
try:
    from psychopy.iohub import launchHubServer
    PSYCHOPY_AVAILABLE = True
except ImportError:
    PSYCHOPY_AVAILABLE = False

class EyeTrackerController:
    """Handles communication with the ABL EyeLink Host PC using PsychoPy ioHub."""
    def __init__(self):
        self.connected = False
        self.tracker = None
        self.io = None
        
    def connect(self):
        if PSYCHOPY_AVAILABLE:
            try:
                # Eye tracker definition exactly as specified in the ABL lab manual
                iohub_tracker_class_path = 'eyetracker.hw.sr_research.eyelink.EyeTracker'
                eyetracker_config = dict()
                eyetracker_config['name'] = 'tracker'
                eyetracker_config['model_name'] = 'EYELINK 1000 DESKTOP'
                eyetracker_config['simulation_mode'] = False
                eyetracker_config['runtime_settings'] = dict(sampling_rate=1000, track_eyes='RIGHT')
                
                # NOTE: The manual sets this to "fname". You might want to dynamically 
                # change this to your participant ID later so files don't overwrite!
                eyetracker_config['default_native_data_file_name'] = "fname" 
                
                # Starting IO hub
                self.io = launchHubServer(**{iohub_tracker_class_path: eyetracker_config})
                self.tracker = self.io.devices.tracker
                
                # At the start of an experiment
                self.tracker.setConnectionState(True)
                self.tracker.setRecordingState(True)
                
                self.connected = True
                print("SUCCESS: Connected to EyeLink Tracker via PsychoPy ioHub.")
            except Exception as e:
                print(f"EyeLink Connection Failed: {e}")
        else:
            print("PsychoPy ioHub not installed. Running in dummy mode for testing.")

    def log_event(self, event_message):
        """Sends a synchronized timestamp trigger to the eye-tracker's data file."""
        if self.connected and self.tracker:
            self.tracker.sendMessage(f"TMT_EVENT: {event_message}")
        else:
            pass 

    def disconnect(self):
        """Safely shuts down tracking at the end of the experiment."""
        if self.connected and self.tracker:
            print("Disconnecting EyeLink...")
            self.tracker.setConnectionState(False)
            self.tracker.setRecordingState(False)
            if self.io:
                self.io.quit()
            self.connected = False

# ---------------------------

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from TMTagent.enviromentTMT import TMTTaskProvider

class SharedLayoutTMTTaskProvider:
    def __init__(self, task_type="A", layout=None):
        self.provider = TMTTaskProvider(task_type=task_type)
        if layout is not None:
            self.provider.nodes = {target: tuple(layout[target]) for target in self.provider.targets}
        self.task_type = self.provider.task_type
        self.targets = self.provider.targets

    def get_current_target(self):
        return self.provider.get_current_target()

    def get_target_coords(self, target):
        return self.provider.get_target_coords(target)

    def submit_action(self, target):
        return self.provider.submit_action(target)

    def get_uncompleted_targets(self):
        return self.provider.get_uncompleted_targets()

    @property
    def current_index(self):
        return self.provider.current_index

    @property
    def completed(self):
        return self.provider.completed

class HumanTMTSession:
    def __init__(self, task_type="A", participant_id="participant", log_path=None, shared_layout=None, eye_tracker=None):
        self.task_type = task_type
        self.participant_id = participant_id
        self.provider = SharedLayoutTMTTaskProvider(task_type=task_type, layout=shared_layout)
        self.eye_tracker = eye_tracker # Pass the tracker in
        self.score = 0
        self.errors = 0
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self.started_timestamp = None
        self.last_correct_timestamp = None
        self.prev_event_ts = None
        self.started = False
        self.log_entries = []

        if log_path is None:
            base_dir = Path(__file__).resolve().parent / "logs"
            base_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_path = base_dir / f"{participant_id}_{task_type}_{stamp}.jsonl"
        else:
            self.log_path = Path(log_path)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._write_event("task_started", {
            "task_type": task_type,
            "targets": self.provider.targets,
            "layout": [list(self.provider.get_target_coords(target)) for target in self.provider.targets],
        })

    def _write_event(self, event_type, payload):
        now_ts = time.time()
        elapsed_since_start_ms = None
        event_delta_ms = None
        
        if self.started_timestamp is not None:
            elapsed_since_start_ms = round((now_ts - self.started_timestamp) * 1000, 3)
        if self.prev_event_ts is not None:
            event_delta_ms = round((now_ts - self.prev_event_ts) * 1000, 3)
        self.prev_event_ts = now_ts

        entry = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "event_type": event_type,
            "task_type": self.task_type,
            "participant_id": self.participant_id,
            "score": self.score,
            "errors": self.errors,
            "current_target": self.provider.get_current_target(),
            "completed_count": self.provider.current_index,
            "completed": self.provider.completed,
            "elapsed_since_start_ms": elapsed_since_start_ms,
            "event_delta_ms": event_delta_ms,
            **payload,
        }
        
        self.log_entries.append(entry)
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")
            
        # --- SEND TRIGGER TO EYELINK ---
        # We don't send mouse moves to avoid flooding the tracker, 
        # but clicks and task boundaries are crucial.
        if self.eye_tracker and event_type != "mouse_move":
            self.eye_tracker.log_event(f"{event_type}_{self.task_type}_Target:{self.provider.get_current_target()}")

    def start_timer(self):
        if not self.started:
            self.started_timestamp = time.time()
            self.started = True
            self._write_event("timer_started", {})

    def log_mouse_move(self, x, y):
        self._write_event("mouse_move", {"x": x, "y": y})

    def submit_click(self, target, x, y):
        expected_target = self.provider.get_current_target()
        if self.provider.completed:
            self._write_event("click_after_completion", {"x": x, "y": y, "target": target})
            return False

        if target == expected_target:
            if self.provider.current_index == 0 and not self.started:
                self.start_timer()

            success = self.provider.submit_action(target)
            if success:
                self.score += 1
                now_ts = time.time()
                correct_interval_ms = None
                if self.last_correct_timestamp is not None:
                    correct_interval_ms = round((now_ts - self.last_correct_timestamp) * 1000, 3)
                self.last_correct_timestamp = now_ts

                self._write_event("correct_click", {
                    "x": x,
                    "y": y,
                    "target": target,
                    "expected_target": expected_target,
                    "correct_interval_ms": correct_interval_ms,
                })
                
                if self.provider.completed:
                    self._write_event("task_completed", {"final_score": self.score, "final_errors": self.errors})
                return True

        self.errors += 1
        self._write_event("incorrect_click", {"x": x, "y": y, "target": target, "expected_target": expected_target})
        return False

    def log_miss_click(self, x, y):
        self.errors += 1
        self._write_event("miss_click", {"x": x, "y": y})

    def finalize(self):
        duration_seconds = None
        if self.started_timestamp is not None:
            duration_seconds = round(time.time() - self.started_timestamp, 3)
        self._write_event("task_summary", {
            "final_score": self.score,
            "final_errors": self.errors,
            "duration_seconds": duration_seconds,
            "total_events": len(self.log_entries),
        })
        
        # Stop the eye tracker recording safely when the task is done
        if self.eye_tracker:
            self.eye_tracker.disconnect()

class HumanTMTApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Human TMT Recorder - Lab Edition")
        
        # --- FULLSCREEN LOGIC ---
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg="#2b2b2b") # Dark background for research focus
        
        # Bind ESC to exit fullscreen safely using the new shutdown method
        self.root.bind("<Escape>", self._safe_exit)

        # Calculate a perfect square canvas based on 85% of screen height
        screen_height = self.root.winfo_screenheight()
        self.canvas_size = int(screen_height * 0.85)
        self.node_radius = int(self.canvas_size * 0.025) # Scale nodes with canvas

        self.participant_name = tk.StringVar(value="participant")
        self.task_label = tk.StringVar(value="TMT-A")
        self.status_text = tk.StringVar(value="Start a task to begin recording")
        self.current_target_text = tk.StringVar(value="Current target: --")

        self.shared_layout = None
        self.task_layouts = {}
        self.sequence_mode = False
        self.session = None
        
        # Initialize Eye Tracker
        self.eye_tracker = EyeTrackerController()
        self.eye_tracker.connect()

        self._build_ui()
        self._initialize_layout()
        self._start_task("A")

    def _build_ui(self):
        style = ttk.Style()
        style.configure("TFrame", background="#2b2b2b")
        style.configure("TLabel", background="#2b2b2b", foreground="white")
        
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="Participant:").pack(side=tk.LEFT)
        participant_entry = ttk.Entry(top_frame, textvariable=self.participant_name, width=15)
        participant_entry.pack(side=tk.LEFT, padx=6)

        ttk.Button(top_frame, text="Start TMT-A", command=lambda: self._start_task("A")).pack(side=tk.LEFT, padx=4)
        ttk.Button(top_frame, text="Start TMT-B", command=lambda: self._start_task("B")).pack(side=tk.LEFT, padx=4)
        ttk.Button(top_frame, text="Run Sequence", command=self._start_sequence).pack(side=tk.LEFT, padx=4)
        ttk.Button(top_frame, text="Reset", command=self._reset_current_task).pack(side=tk.LEFT, padx=4)
        
        # Safe exit button
        ttk.Button(top_frame, text="Exit (ESC)", command=self._safe_exit).pack(side=tk.RIGHT)

        info_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        info_frame.pack(fill=tk.X)
        ttk.Label(info_frame, textvariable=self.task_label, font=("Segoe UI", 14, "bold")).pack(side=tk.LEFT)
        ttk.Label(info_frame, textvariable=self.current_target_text, font=("Segoe UI", 12)).pack(side=tk.LEFT, padx=16)
        ttk.Label(info_frame, textvariable=self.status_text, foreground="#4da6ff", font=("Segoe UI", 12)).pack(side=tk.LEFT, padx=16)

        # Center the canvas using expand=True
        canvas_container = tk.Frame(self.root, bg="#2b2b2b")
        canvas_container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_container, width=self.canvas_size, height=self.canvas_size, bg="#f5f5f5", highlightthickness=0)
        # Anchor canvas in the center
        self.canvas.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        self.canvas.bind("<Motion>", self._handle_mouse_move)
        self.canvas.bind("<Button-1>", self._handle_mouse_click)

    def _initialize_layout(self):
        base_nodes = TMTTaskProvider(task_type="A").nodes
        ordered_positions = [base_nodes[str(i)] for i in range(1, 26)]

        tmt_a_layout = {str(i): ordered_positions[i - 1] for i in range(1, 26)}
        tmt_b_targets = ["1", "A", "2", "B", "3", "C", "4", "D", "5", "E", "6", "F", "7", "G", "8", "H", "9", "I", "10", "J", "11", "K", "12", "L", "13"]
        tmt_b_layout = {tmt_b_targets[idx]: ordered_positions[idx] for idx in range(len(tmt_b_targets))}

        self.shared_layout = {
            "A": tmt_a_layout,
            "B": tmt_b_layout,
        }

    def _start_sequence(self):
        self.sequence_mode = True
        self._start_task("A")
        self.status_text.set("Sequence mode: complete TMT-A, then TMT-B will start automatically.")

    def _start_task(self, task_type):
        if self.session is not None and not self.session.provider.completed:
            self.session.finalize()

        self.session = HumanTMTSession(
            task_type=task_type,
            participant_id=self.participant_name.get().strip() or "participant",
            shared_layout=self.shared_layout[task_type],
            eye_tracker=self.eye_tracker 
        )
        self.task_label.set(f"TMT-{task_type}")
        self.current_target_text.set(f"Current target: {self.session.provider.get_current_target()}")
        self.status_text.set("Click the next target node in the correct sequence")
        self._redraw_canvas()

    def _reset_current_task(self):
        if self.session is None:
            return
        task_type = self.session.task_type
        self._start_task(task_type)

    def _redraw_canvas(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, self.canvas_size, self.canvas_size, fill="#f8f8f8", outline="#eeeeee")

        provider = self.session.provider
        completed_count = provider.current_index
        current_target = provider.get_current_target()

        padding = self.canvas_size * 0.08 # Keep nodes away from edges
        active_area = self.canvas_size - (padding * 2)

        for idx, target in enumerate(provider.targets):
            x, y = provider.get_target_coords(target)
            
            # Map the 0-100 coordinates to the new active area
            canvas_x = int((x / 100.0) * active_area + padding)
            canvas_y = int((y / 100.0) * active_area + padding)

            is_completed = idx < completed_count
            if is_completed:
                color = "#90ee90"
                outline = "#2e8b57"
                width = 2
            else:
                color = "#ffffff"
                outline = "#333333"
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

        self.canvas.create_text(
            20,
            20,
            anchor="nw",
            text=f"Score: {self.session.score}   Errors: {self.session.errors}   Next: {current_target or 'Complete'}",
            font=("Segoe UI", 12, "bold"),
            fill="#1b1b1b",
        )

    def _handle_mouse_move(self, event):
        if self.session is None:
            return
        self.session.log_mouse_move(event.x, event.y)

    def _handle_mouse_click(self, event):
        if self.session is None:
            return

        hit_target = self._target_at(event.x, event.y)
        if hit_target is None:
            self.session.log_miss_click(event.x, event.y)
            self.status_text.set("Missed click. Try the next target node.")
        else:
            self.session.submit_click(hit_target, event.x, event.y)
            self.status_text.set(f"Clicked {hit_target}")

        self.current_target_text.set(f"Current target: {self.session.provider.get_current_target() or 'Complete'}")
        self._redraw_canvas()

        if self.session.provider.completed:
            self.session.finalize()
            if self.sequence_mode and self.session.task_type == "A":
                self.status_text.set("TMT-A complete. Starting TMT-B...")
                self.root.after(500, lambda: self._start_task("B"))
            elif self.sequence_mode and self.session.task_type == "B":
                self.status_text.set("TMT-B complete. Sequence finished.")
                self.sequence_mode = False
            else:
                self.status_text.set("Task complete. You can start another task.")

    def _target_at(self, x, y):
        if self.session is None:
            return None
        provider = self.session.provider
        
        padding = self.canvas_size * 0.08
        active_area = self.canvas_size - (padding * 2)

        for target in provider.targets:
            node_x, node_y = provider.get_target_coords(target)
            
            canvas_x = int((node_x / 100.0) * active_area + padding)
            canvas_y = int((node_y / 100.0) * active_area + padding)
            
            distance = ((x - canvas_x) ** 2 + (y - canvas_y) ** 2) ** 0.5
            if distance <= self.node_radius + 4: # Small generous hit-box
                return target
        return None
        
    def _safe_exit(self, event=None):
        """Gracefully shut down the tracker and finalize logs before closing."""
        self.status_text.set("Saving data and disconnecting EyeLink...")
        self.root.update() # Force UI to show the message so you know it's working
        
        # Finalize the session if you exit mid-task
        if self.session is not None and not self.session.provider.completed:
            self.session.finalize()
        # If no session is active but the tracker is connected, disconnect it
        elif self.eye_tracker and self.eye_tracker.connected:
            self.eye_tracker.disconnect()
            
        # Now it is safe to close the window
        self.root.destroy()

    def run(self):
        self.root.mainloop()

def main():
    root = tk.Tk()
    app = HumanTMTApp(root)
    app.run()

if __name__ == "__main__":
    main()