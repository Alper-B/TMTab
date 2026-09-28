import json
import math
import random
from pathlib import Path
from datetime import datetime

try:
    from enviromentTMT import TMTTaskProvider
except ImportError:
    print("Error: Could not import TMTTaskProvider.")
    exit(1)

class CognitiveMetrics:
    def __init__(self, task_type, dynamic_params=None):
        self.task_type = task_type
        
        if dynamic_params is None:
            dynamic_params = {
                "Memory (ms)": 60.0, "Search (ms)": 1100.0, "Motor (ms)": 400.0, 
                "Total Skips": 15, "Saccades/Search": 8.0,
                "Memory Capacity": 2, "Memory Noise (deg)": 1.2
            }

        self.memory_speed = float(dynamic_params.get("Memory (ms)", 60.0))
        self.search_speed = float(dynamic_params.get("Search (ms)", 1100.0))
        self.motor_speed = float(dynamic_params.get("Motor (ms)", 400.0))
        self.total_skips = int(dynamic_params.get("Total Skips", 15))
        self.saccades_per_target = float(dynamic_params.get("Saccades/Search", 8.0))
        self.memory_capacity = int(dynamic_params.get("Memory Capacity", 2))
        self.memory_noise_deg = float(dynamic_params.get("Memory Noise (deg)", 1.2))

class CognitiveTMTAgent:
    def __init__(self, task_type="A", participant_id="RL_Imitation_Bot", custom_metrics=None, screen_dist_mm=600.0, pixel_pitch_mm=0.27):
        self.task_type = task_type
        self.participant_id = participant_id
        self.metrics = CognitiveMetrics(task_type, custom_metrics)
        
        self.screen_dist_mm = screen_dist_mm
        self.pixel_pitch_mm = pixel_pitch_mm
        
        self.current_time_ms = 0
        self.asc_lines = []
        self.json_events = []
        self.memory_bank = {}
        
        # Radii in pixels: Radius = Diameter / 2
        self.fovea_radius_px = self.degrees_to_pixels(2.0) / 2.0
        self.peripheral_radius_px = self.degrees_to_pixels(5.0) / 2.0
        self.memory_noise_px = self.degrees_to_pixels(self.metrics.memory_noise_deg) / 2.0
        
        self.screen_w, self.screen_h = 1920, 1080
        self.canvas_size = int(self.screen_h * 0.85)
        self.offset_x = (self.screen_w - self.canvas_size) / 2
        self.offset_y = (self.screen_h - self.canvas_size) / 2
        
        self.provider = TMTTaskProvider(task_type=task_type)
        self.targets = self.provider.targets
        
        self.gaze_x, self.gaze_y = self._get_screen_coords(self.targets[0])
        self.mouse_x, self.mouse_y = self.gaze_x, self.gaze_y

    def degrees_to_pixels(self, deg):
        if self.screen_dist_mm <= 0 or self.pixel_pitch_mm <= 0: return 0.0
        physical_mm = 2.0 * self.screen_dist_mm * math.tan(math.radians(deg / 2.0))
        return physical_mm / self.pixel_pitch_mm

    def pixels_to_degrees(self, px):
        if self.screen_dist_mm <= 0 or self.pixel_pitch_mm <= 0: return 0.0
        physical_mm = px * self.pixel_pitch_mm
        return math.degrees(2.0 * math.atan(physical_mm / (2.0 * self.screen_dist_mm)))

    def _get_screen_coords(self, target):
        pct_x, pct_y = self.provider.get_target_coords(target)
        padding = self.canvas_size * 0.08
        active = self.canvas_size - (padding * 2.0)
        return (pct_x / 100.0) * active + padding + self.offset_x, (pct_y / 100.0) * active + padding + self.offset_y

    def _write_json_event(self, event_type, payload):
        self.json_events.append({
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "event_type": event_type, "task_type": self.task_type,
            "participant_id": self.participant_id, "score": self.provider.current_index,
            "completed_count": self.provider.current_index, "elapsed_since_start_ms": self.current_time_ms,
            **payload
        })

    def _log_asc_sample(self):
        jitter = random.uniform(-0.4, 0.4)
        self.asc_lines.append(f"{self.current_time_ms}\t{self.gaze_x + jitter:6.1f}\t{self.gaze_y + jitter:6.1f}\t335.0\t32768.0\t...")

    def _log_asc_msg(self, msg):
        self.asc_lines.append(f"MSG\t{self.current_time_ms} TMT_EVENT: {msg}")

    def _execute_fixation(self, duration_ms, drift_mouse=False):
        start_time = self.current_time_ms
        self.asc_lines.append(f"SFIX R   {start_time}")
        steps = max(1, int(duration_ms))
        
        for i in range(1, steps + 1):
            if drift_mouse:
                t = i / steps
                self.mouse_x += (self.gaze_x - self.mouse_x) * (t * 0.04)
                self.mouse_y += (self.gaze_y - self.mouse_y) * (t * 0.04)
            self._log_asc_sample()
            self.current_time_ms += 1
            if drift_mouse and self.current_time_ms % 10 == 0:
                self._write_json_event("mouse_move", {"x": self.mouse_x, "y": self.mouse_y})
                
        self.asc_lines.append(f"EFIX R   {start_time}\t{self.current_time_ms - 1}\t{steps}\t{self.gaze_x:6.1f}\t{self.gaze_y:6.1f}\t    300")

    def _execute_saccade(self, target_x, target_y, duration_ms):
        start_time = self.current_time_ms
        start_x, start_y = self.gaze_x, self.gaze_y
        self.asc_lines.append(f"SSACC R  {start_time}")
        steps = max(1, int(duration_ms))
        
        for i in range(1, steps + 1):
            t = i / steps
            smooth_t = t * t * (3.0 - 2.0 * t) 
            self.gaze_x = start_x + (target_x - start_x) * smooth_t
            self.gaze_y = start_y + (target_y - start_y) * smooth_t
            self._log_asc_sample()
            self.current_time_ms += 1
            
        dist = math.hypot(target_x - start_x, target_y - start_y)
        deg_dist = self.pixels_to_degrees(dist)
        vel_deg_ms = deg_dist / steps if steps > 0 else 0
        self.asc_lines.append(f"ESACC R  {start_time}\t{self.current_time_ms - 1}\t{steps}\t{start_x:6.1f}\t{start_y:6.1f}\t{target_x:6.1f}\t{target_y:6.1f}\t{vel_deg_ms:6.2f}\t    300")

    def _execute_mouse_move(self, target_x, target_y, duration_ms):
        start_time = self.current_time_ms
        self.asc_lines.append(f"SFIX R   {start_time}")
        start_x, start_y = self.mouse_x, self.mouse_y
        steps = max(1, int(duration_ms))
        
        for i in range(1, steps + 1):
            t = i / steps
            smooth_t = t * t * (3.0 - 2.0 * t)
            self.mouse_x = start_x + (target_x - start_x) * smooth_t
            self.mouse_y = start_y + (target_y - start_y) * smooth_t
            self._log_asc_sample()
            self.current_time_ms += 1
            if self.current_time_ms % 10 == 0:
                self._write_json_event("mouse_move", {"x": self.mouse_x, "y": self.mouse_y})
                
        self.mouse_x, self.mouse_y = float(target_x), float(target_y)
        self._write_json_event("mouse_move", {"x": self.mouse_x, "y": self.mouse_y})
        self.asc_lines.append(f"EFIX R   {start_time}\t{self.current_time_ms - 1}\t{steps}\t{self.gaze_x:6.1f}\t{self.gaze_y:6.1f}\t    300")

    def _scan_periphery(self, uncompleted, current_target):
        if self.metrics.memory_capacity <= 0: return
        for tgt in uncompleted:
            if tgt == current_target or tgt in self.memory_bank: continue
            tx, ty = self._get_screen_coords(tgt)
            if math.hypot(tx - self.gaze_x, ty - self.gaze_y) < self.peripheral_radius_px:
                noise_x = random.gauss(0, self.memory_noise_px)
                noise_y = random.gauss(0, self.memory_noise_px)
                self.memory_bank[tgt] = (tx + noise_x, ty + noise_y)
                if len(self.memory_bank) > self.metrics.memory_capacity:
                    del self.memory_bank[next(iter(self.memory_bank))]

    def run_simulation(self):
        self.asc_lines, self.json_events, self.current_time_ms = [], [], 0
        self.memory_bank.clear()
        
        self._write_json_event("task_started", {
            "targets": self.targets,
            "layout": [list(self.provider.get_target_coords(t)) for t in self.targets],
            "physical_setup": {
                "distance_to_screen_mm": self.screen_dist_mm,
                "monitor_pixel_pitch_mm": self.pixel_pitch_mm
            }
        })
        self._log_asc_msg("timer_started")
        
        first_target = self.targets[0]
        self.mouse_x, self.mouse_y = self._get_screen_coords(first_target)
        self.provider.submit_action(first_target)
        self._write_json_event("correct_click", {"target": first_target, "x": self.mouse_x, "y": self.mouse_y, "current_target": first_target})
        self._log_asc_msg(f"correct_click_{self.task_type}_Target:{first_target}")

        for i in range(1, len(self.targets)):
            current_target = self.targets[i]
            t_x, t_y = self._get_screen_coords(current_target)
            uncompleted = self.provider.get_uncompleted_targets()
            
            # 1. Memory / Planning Dwell
            mem_time = self.metrics.memory_speed * random.uniform(0.9, 1.1)
            self._execute_fixation(mem_time)
            
            # 2. Search vs Memory Jump
            if current_target in self.memory_bank:
                rem_x, rem_y = self.memory_bank.pop(current_target)
                self._execute_saccade(rem_x, rem_y, duration_ms=random.randint(25, 45))
                self._execute_fixation(random.uniform(90, 130))
                # Foveal corrective micro-saccade
                self._execute_saccade(t_x, t_y, duration_ms=random.randint(15, 25))
                self._execute_fixation(random.uniform(140, 220))
            else:
                saccade_target_count = max(1, int(random.gauss(self.metrics.saccades_per_target, 1.5)))
                time_per_saccade = self.metrics.search_speed / max(1, saccade_target_count)
                
                sorted_pool = sorted(uncompleted, key=lambda n: math.hypot(self._get_screen_coords(n)[0] - self.gaze_x, self._get_screen_coords(n)[1] - self.gaze_y))
                candidate_pool = sorted_pool[:max(3, len(sorted_pool) // 3)]
                
                for s in range(saccade_target_count):
                    if s == saccade_target_count - 1:
                        self._execute_saccade(t_x, t_y, duration_ms=random.randint(25, 45))
                        self._execute_fixation(random.uniform(150, 240))
                    else:
                        dx, dy = self._get_screen_coords(random.choice(candidate_pool))
                        dx += random.uniform(-40, 40)
                        dy += random.uniform(-40, 40)
                        self._execute_saccade(dx, dy, duration_ms=random.randint(20, 38))
                        self._scan_periphery(uncompleted, current_target)
                        self._execute_fixation(time_per_saccade * random.uniform(0.75, 1.15), drift_mouse=True)

            self.gaze_x, self.gaze_y = t_x, t_y
            
            # 3. Motor Movement
            mot_time = self.metrics.motor_speed * random.uniform(0.85, 1.15)
            self._execute_mouse_move(t_x, t_y, mot_time)
            
            self._write_json_event("correct_click", {"target": current_target, "x": self.mouse_x, "y": self.mouse_y, "current_target": current_target})
            self._log_asc_msg(f"correct_click_{self.task_type}_Target:{current_target}")
            self.provider.submit_action(current_target)
            
        self._write_json_event("task_completed", {"final_score": 25})
        return self._save_files()

    def _save_files(self):
        base_dir = Path("sim_logs")
        base_dir.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        asc_path = base_dir / f"{self.participant_id}_{self.task_type}_{stamp}.asc"
        with open(asc_path, "w", encoding="utf-8") as f: f.write("\n".join(self.asc_lines))
        json_path = base_dir / f"{self.participant_id}_{self.task_type}_{stamp}.jsonl"
        with open(json_path, "w", encoding="utf-8") as f:
            for ev in self.json_events: f.write(json.dumps(ev) + "\n")
        return str(json_path), str(asc_path)