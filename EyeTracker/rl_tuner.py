import json
import math
from pathlib import Path
import numpy as np

from agent import CognitiveTMTAgent
from visualizer import TMTSession, HeadlessCognitiveAnalyzer

class TrajectoryRewardPolicyTuner:
    """Tunes agent cognitive parameters using behavioral imitation loss against human recording."""
    def __init__(self, human_session, canvas_size=918):
        self.human_session = human_session
        self.task_type = human_session.task_type
        self.canvas_size = canvas_size
        self.analyzer = HeadlessCognitiveAnalyzer(canvas_size=canvas_size)
        
        # Extract ground truth human trajectory milestones
        self.human_clicks = [ev for ev in human_session.events if ev.get("event_type") == "correct_click"]
        self.human_metrics = self.analyzer.analyze_session(human_session)

    def compute_behavioral_loss(self, policy_vector):
        """Calculates trajectory error, saccade count delta, and cognitive metric divergence."""
        mem_dwell, search_time, motor_time, saccades_per_tgt, capacity, noise_deg = policy_vector
        
        params = {
            "Memory (ms)": mem_dwell,
            "Search (ms)": search_time,
            "Motor (ms)": motor_time,
            "Saccades/Search": saccades_per_tgt,
            "Memory Capacity": int(round(capacity)),
            "Memory Noise (deg)": noise_deg
        }
        
        agent = CognitiveTMTAgent(
            task_type=self.task_type,
            participant_id="RL_Probe",
            custom_metrics=params,
            screen_dist_mm=self.human_session.screen_dist_mm,
            pixel_pitch_mm=self.human_session.pixel_pitch_mm
        )
        sim_json, sim_asc = agent.run_simulation()
        
        sim_session = TMTSession(self.task_type)
        sim_session.load_jsonl(sim_json)
        sim_session.load_asc(sim_asc)
        sim_metrics = self.analyzer.analyze_session(sim_session)
        
        # 1. Timeline Synchronization Loss (Per-Target Click Delays)
        sim_clicks = [ev for ev in sim_session.events if ev.get("event_type") == "correct_click"]
        time_diffs = []
        for h_c, s_c in zip(self.human_clicks, sim_clicks):
            h_t = h_c.get("elapsed_since_start_ms", 0.0) or 0.0
            s_t = s_c.get("elapsed_since_start_ms", 0.0) or 0.0
            time_diffs.append((h_t - s_t) ** 2)
        timeline_loss = math.sqrt(np.mean(time_diffs)) / max(1.0, self.human_session.max_time_ms)
        
        # 2. Cognitive Metric Delta Loss
        metric_keys = ["Memory (ms)", "Search (ms)", "Motor (ms)", "Saccades/Search", "Memory Noise (deg)"]
        metric_errors = []
        for k in metric_keys:
            norm = max(self.human_metrics.get(k, 1.0), 0.1)
            err = ((self.human_metrics.get(k, 0.0) - sim_metrics.get(k, 0.0)) / norm) ** 2
            metric_errors.append(err)
        cog_loss = np.mean(metric_errors)
        
        # Cleanup probe logs
        Path(sim_json).unlink(missing_ok=True)
        Path(sim_asc).unlink(missing_ok=True)
        
        total_loss = 0.6 * timeline_loss + 0.4 * cog_loss
        return total_loss, sim_metrics

    def fit_parameters(self, max_episodes=25, population_size=6, callback=None):
        """Cross-Entropy Policy Search over biological parameter space."""
        # Mean priors initialized to human algorithmic estimates
        mu = np.array([
            max(20.0, self.human_metrics.get("Memory (ms)", 50.0)),
            max(300.0, self.human_metrics.get("Search (ms)", 1000.0)),
            max(150.0, self.human_metrics.get("Motor (ms)", 400.0)),
            max(2.0, self.human_metrics.get("Saccades/Search", 6.0)),
            float(max(1, self.human_metrics.get("Memory Capacity", 2))),
            max(0.5, self.human_metrics.get("Memory Noise (deg)", 1.5))
        ], dtype=float)
        
        sigma = np.array([20.0, 250.0, 80.0, 2.5, 0.8, 0.4], dtype=float)
        
        bounds_low = np.array([15.0, 250.0, 100.0, 1.0, 1.0, 0.3])
        bounds_high = np.array([250.0, 3500.0, 900.0, 25.0, 5.0, 4.0])
        
        best_loss = float('inf')
        best_params = None
        best_metrics = None
        
        for ep in range(max_episodes):
            candidates = np.random.normal(mu, sigma, size=(population_size, 6))
            candidates = np.clip(candidates, bounds_low, bounds_high)
            
            results = []
            for cand in candidates:
                loss, m = self.compute_behavioral_loss(cand)
                results.append((loss, cand, m))
                if loss < best_loss:
                    best_loss = loss
                    best_params = cand
                    best_metrics = m
            
            results.sort(key=lambda x: x[0])
            elite = np.array([x[1] for x in results[:max(2, population_size // 3)]])
            
            mu = 0.7 * mu + 0.3 * np.mean(elite, axis=0)
            sigma = 0.7 * sigma + 0.3 * np.std(elite, axis=0) + 1e-4
            
            if callback:
                callback(ep + 1, max_episodes, best_loss, best_params)
                
        return {
            "Memory (ms)": float(best_params[0]),
            "Search (ms)": float(best_params[1]),
            "Motor (ms)": float(best_params[2]),
            "Saccades/Search": float(best_params[3]),
            "Memory Capacity": int(round(best_params[4])),
            "Memory Noise (deg)": float(best_params[5]),
            "Fitting Loss": float(best_loss)
        }