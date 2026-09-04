import json
import math
import random
from pathlib import Path
from datetime import datetime

try:
    from enviromentTMT import TMTTaskProvider
except ImportError:
    print("Error: Could not import TMTTaskProvider. Ensure TMTagent package is in your path.")
    exit(1)

class CognitiveMetrics:
    def __init__(self, task_type):
        self.task_type = task_type
        if task_type == "A":
            self.memory_speed = 48
            self.search_speed = 988
            self.motor_speed = 417
            self.total_skips = 19
            self.total_saccades = 214.2
        else:
            self.search_speed = 1876
            self.memory_speed = 78
            self.motor_speed = 510
            self.total_skips = 61
            self.total_saccades = 438.5

        self.skips_per_target = self.total_skips / 24.0
        self.saccades_per_target = self.total_saccades / 24.0

class CognitiveTMTAgent:
    def __init__(self, task_type="A", participant_id="Synthetic_Bot"):
        self.task_type = task_type
        self.participant_id = participant_id
        self.metrics = CognitiveMetrics(task_type)
        
        self.current_time_ms = 0
        self.asc_lines = []
        self.json_events = []
        
        self.screen_w = 1920
        self.screen_h = 1080
        self.canvas_size = int(self.screen_h * 0.85)
        self.offset_x = (self.screen_w - self.canvas_size) / 2
        self.offset_y = (self.screen_h - self.canvas_size) / 2
        
        self.provider = TMTTaskProvider(task_type=task_type)
        self.targets = self.provider.targets
        
        # Explicitly align starting position to exact center of Target 1
        self.gaze_x, self.gaze_y = self._get_screen_coords(self.targets[0])
        self.mouse_x, self.mouse_y = self.gaze_x, self.gaze_y
        
    def _get_screen_coords(self, target):
        """Translates percentages to absolute pixel coordinates."""
        pct_x, pct_y = self.provider.get_target_coords(target)
        padding = self.canvas_size * 0.08
        active = self.canvas_size - (padding * 2)
        cx = (pct_x / 100.0) * active + padding
        cy = (pct_y / 100.0) * active + padding
        return cx + self.offset_x, cy + self.offset_y

    def _write_json_event(self, event_type, payload):
        self.json_events.append({
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "event_type": event_type,
            "task_type": self.task_type,
            "participant_id": self.participant_id,
            "score": self.provider.current_index,
            "completed_count": self.provider.current_index,
            "elapsed_since_start_ms": self.current_time_ms,
            **payload
        })

    def _log_asc_sample(self):
        jitter_x = random.uniform(-0.5, 0.5)
        jitter_y = random.uniform(-0.5, 0.5)
        pupil = random.uniform(330.0, 340.0)
        self.asc_lines.append(f"{self.current_time_ms}\t{self.gaze_x + jitter_x:6.1f}\t{self.gaze_y + jitter_y:6.1f}\t{pupil:6.1f}\t32768.0\t...")

    def _log_asc_msg(self, msg):
        self.asc_lines.append(f"MSG\t{self.current_time_ms} TMT_EVENT: {msg}")

    def _execute_fixation(self, duration_ms, drift_mouse=False):
        start_time = self.current_time_ms
        self.asc_lines.append(f"SFIX R   {start_time}")
        
        steps = int(duration_ms)
        mx_start, my_start = self.mouse_x, self.mouse_y
        
        for i in range(1, steps + 1):
            if drift_mouse:
                t = i / steps
                self.mouse_x = mx_start + (self.gaze_x - mx_start) * (t * 0.05)
                self.mouse_y = my_start + (self.gaze_y - my_start) * (t * 0.05)
                
            self._log_asc_sample()
            self.current_time_ms += 1
            
            if drift_mouse and self.current_time_ms % 10 == 0:
                self._write_json_event("mouse_move", {"x": self.mouse_x, "y": self.mouse_y})
                
        end_time = self.current_time_ms - 1
        self.asc_lines.append(f"EFIX R   {start_time}\t{end_time}\t{int(duration_ms)}\t{self.gaze_x:6.1f}\t{self.gaze_y:6.1f}\t    300")

    def _execute_saccade(self, target_x, target_y, duration_ms):
        start_time = self.current_time_ms
        start_x, start_y = self.gaze_x, self.gaze_y
        self.asc_lines.append(f"SSACC R  {start_time}")
        
        steps = int(duration_ms)
        for i in range(1, steps + 1):
            t = i / steps
            smooth_t = t * t * (3 - 2 * t) 
            self.gaze_x = start_x + (target_x - start_x) * smooth_t
            self.gaze_y = start_y + (target_y - start_y) * smooth_t
            self._log_asc_sample()
            self.current_time_ms += 1
            
        end_time = self.current_time_ms - 1
        dist = math.hypot(target_x - start_x, target_y - start_y)
        velocity = dist / duration_ms if duration_ms > 0 else 0
        self.asc_lines.append(f"ESACC R  {start_time}\t{end_time}\t{steps}\t{start_x:6.1f}\t{start_y:6.1f}\t{target_x:6.1f}\t{target_y:6.1f}\t{velocity:6.2f}\t    300")

    def _execute_mouse_move(self, target_x, target_y, duration_ms):
        start_time = self.current_time_ms
        self.asc_lines.append(f"SFIX R   {start_time}")
        
        start_x, start_y = self.mouse_x, self.mouse_y
        steps = int(duration_ms)
        
        for i in range(1, steps + 1):
            t = i / steps
            smooth_t = t * t * (3 - 2 * t)
            self.mouse_x = start_x + (target_x - start_x) * smooth_t
            self.mouse_y = start_y + (target_y - start_y) * smooth_t
            
            self._log_asc_sample()
            self.current_time_ms += 1
            
            if self.current_time_ms % 10 == 0:
                self._write_json_event("mouse_move", {"x": self.mouse_x, "y": self.mouse_y})

        # CALIBRATION ENFORCEMENT: Eliminate floating-point drift at the end of the move
        self.mouse_x = float(target_x)
        self.mouse_y = float(target_y)
        self._write_json_event("mouse_move", {"x": self.mouse_x, "y": self.mouse_y})

        end_time = self.current_time_ms - 1
        self.asc_lines.append(f"EFIX R   {start_time}\t{end_time}\t{int(duration_ms)}\t{self.gaze_x:6.1f}\t{self.gaze_y:6.1f}\t    300")

    def run_simulation(self):
        print(f"--- Booting Agent for TMT-{self.task_type} ---")
        
        self.asc_lines = []
        self.json_events = []
        self.current_time_ms = 0
        
        self._write_json_event("task_started", {
            "targets": self.targets,
            "layout": [list(self.provider.get_target_coords(t)) for t in self.targets]
        })
        self._log_asc_msg("timer_started")
        
        first_target = self.targets[0]
        # Guarantee dead-center start before the very first click
        self.mouse_x, self.mouse_y = self._get_screen_coords(first_target)
        self.provider.submit_action(first_target)
        self._write_json_event("correct_click", {"target": first_target, "x": self.mouse_x, "y": self.mouse_y, "current_target": first_target})
        self._log_asc_msg(f"correct_click_{self.task_type}_Target:{first_target}")
        print(f"[{self.current_time_ms}ms] Agent clicked {first_target} (START)")

        for i in range(1, len(self.targets)):
            current_target = self.targets[i]
            t_x, t_y = self._get_screen_coords(current_target)
            
            mem_time = self.metrics.memory_speed * random.uniform(0.85, 1.15) 
            self._execute_fixation(mem_time)
            
            actual_saccades = max(1, int(random.gauss(self.metrics.saccades_per_target, 2)))
            time_per_saccade_cycle = self.metrics.search_speed / max(1, actual_saccades)
            
            remaining_nodes = self.provider.get_uncompleted_targets()
            sorted_nodes = sorted(remaining_nodes, key=lambda n: math.hypot(self._get_screen_coords(n)[0] - self.gaze_x, self._get_screen_coords(n)[1] - self.gaze_y))
            search_pool = sorted_nodes[:max(3, len(sorted_nodes)//3)] 
            
            for s in range(actual_saccades):
                if s == actual_saccades - 1:
                    self._execute_saccade(t_x, t_y, duration_ms=random.randint(25, 45))
                    registration_time = random.uniform(150, 250)
                    self._execute_fixation(registration_time)
                else:
                    dist_coords = self._get_screen_coords(random.choice(search_pool))
                    dx = dist_coords[0] + random.uniform(-50, 50)
                    dy = dist_coords[1] + random.uniform(-50, 50)
                    
                    self._execute_saccade(dx, dy, duration_ms=random.randint(20, 40))
                    self._execute_fixation(duration_ms=time_per_saccade_cycle * random.uniform(0.7, 1.1), drift_mouse=True)

            self.gaze_x, self.gaze_y = t_x, t_y
            
            mot_time = self.metrics.motor_speed * random.uniform(0.8, 1.2)
            self._execute_mouse_move(t_x, t_y, mot_time)
            
            self._write_json_event("correct_click", {"target": current_target, "x": self.mouse_x, "y": self.mouse_y, "current_target": current_target})
            self._log_asc_msg(f"correct_click_{self.task_type}_Target:{current_target}")
            self.provider.submit_action(current_target)
            print(f"[{self.current_time_ms}ms] Agent clicked {current_target}")
            
        self._write_json_event("task_completed", {"final_score": 25})
        self._save_files()

    def _save_files(self):
        base_dir = Path("sim_logs")
        base_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        asc_path = base_dir / f"{self.participant_id}_{self.task_type}_{stamp}.asc"
        with open(asc_path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.asc_lines))
            
        json_path = base_dir / f"{self.participant_id}_{self.task_type}_{stamp}.jsonl"
        with open(json_path, "w", encoding="utf-8") as f:
            for ev in self.json_events:
                f.write(json.dumps(ev) + "\n")
                
        print(f"Simulation Complete. Logs saved to: {base_dir.resolve()}")

if __name__ == "__main__":
    agent_a = CognitiveTMTAgent(task_type="A", participant_id="Bot_Refactored")
    agent_a.run_simulation()
    
    agent_b = CognitiveTMTAgent(task_type="B", participant_id="Bot_Refactored")
    agent_b.run_simulation()