import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class TMTTarget:
    label: str


class TMTTaskProvider:
    """Minimal environment that serves targets and measures agent actions."""

    def __init__(self, mode: str, targets: List[str]):
        self.mode = mode
        self.targets = [TMTTarget(label=t) for t in targets]
        self.current_index = 0
        self.history: List[Dict[str, Any]] = []
        self.total_measured_ms = 0.0
        self.round_count = 0

    def reset(self) -> None:
        self.current_index = 0
        self.history = []
        self.total_measured_ms = 0.0
        self.round_count = 0

    def get_current_target(self) -> TMTTarget:
        return self.targets[self.current_index]

    def submit_action(self, action_label: str, step_name: str, planned_latency_ms: float) -> bool:
        target = self.get_current_target()
        measured_ms = round(max(8.0, planned_latency_ms + random.uniform(4.0, 12.0)), 2)
        success = action_label == target.label

        self.history.append(
            {
                "step": step_name,
                "target": target.label,
                "action": action_label,
                "measured_ms": measured_ms,
                "success": success,
            }
        )
        self.total_measured_ms += measured_ms
        self.round_count += 1

        if success:
            self.current_index += 1
            print(
                f"[ENV] {self.mode} | ACTION OK | step={step_name:<26} target={target.label} measured={measured_ms:>6.2f}ms"
            )
        else:
            print(
                f"[ENV] {self.mode} | ACTION FAIL | step={step_name:<26} expected={target.label} got={action_label} measured={measured_ms:>6.2f}ms"
            )
        return success

    def log_cognitive_step(self, step_name: str, latency_ms: float, detail: str = "") -> None:
        print(f"[COG] {self.mode} | step={step_name:<26} latency={latency_ms:>6.2f}ms {detail}".rstrip())


class CognitiveAgent:
    """Heuristic agent that follows the requested TMT flowchart steps."""

    def __init__(self, provider: TMTTaskProvider):
        self.provider = provider
        self.rng = random.Random(7)
        self.flowchart = self._flowchart_for_mode(provider.mode)
        self.latency_baseline_ms = {
            "Goal Retrieval": 95,
            "Visual Search": 115,
            "Target Identification": 88,
            "Visuomotor Planning": 72,
            "Motor Execution": 104,
            "Performance Monitoring": 69,
            "Update Working Memory": 58,
            "Maintain Alternation Rule": 82,
            "Inhibit Previous Set": 74,
            "Switch Cognitive Set": 90,
            "Visuomotor Planning & Motor Execution": 128,
        }

    def _flowchart_for_mode(self, mode: str) -> List[str]:
        if mode == "A":
            return [
                "Goal Retrieval",
                "Visual Search",
                "Target Identification",
                "Visuomotor Planning",
                "Motor Execution",
                "Performance Monitoring",
                "Update Working Memory",
            ]
        return [
            "Goal Retrieval",
            "Maintain Alternation Rule",
            "Inhibit Previous Set",
            "Switch Cognitive Set",
            "Visual Search",
            "Target Identification",
            "Visuomotor Planning & Motor Execution",
            "Performance Monitoring",
            "Update Working Memory",
        ]

    def _sample_latency(self, step_name: str) -> float:
        base = self.latency_baseline_ms.get(step_name, 100)
        jitter = self.rng.uniform(-0.12 * base, 0.12 * base)
        return round(base + jitter, 2)

    def run(self) -> None:
        print(f"\n=== Running TMT-{self.provider.mode} ===")
        print("The agent follows the requested cognitive flowchart without any RL or heavy abstraction.")

        while self.provider.current_index < len(self.provider.targets):
            current_target = self.provider.get_current_target()
            print(f"\n[ROUND] target={current_target.label} | remaining={len(self.provider.targets) - self.provider.current_index}")

            for step_name in self.flowchart:
                latency_ms = self._sample_latency(step_name)
                self.provider.log_cognitive_step(step_name, latency_ms, detail=f"target={current_target.label}")

                if step_name in {"Motor Execution", "Visuomotor Planning & Motor Execution"}:
                    self.provider.submit_action(current_target.label, step_name, latency_ms)
                elif step_name in {"Visual Search", "Target Identification"}:
                    self.provider.submit_action(current_target.label, step_name, latency_ms)
                else:
                    time.sleep(0.005)

                if self.provider.current_index >= len(self.provider.targets):
                    break

            if self.provider.current_index < len(self.provider.targets):
                self.provider.log_cognitive_step(
                    "Update Working Memory",
                    self._sample_latency("Update Working Memory"),
                    detail="commit next target",
                )

        print("\nSimulation complete.")
        print(f"Total measured environment time: {self.provider.total_measured_ms:.2f}ms")
        print("Per-step history:")
        for entry in self.provider.history:
            print(
                f"  - step={entry['step']:<26} target={entry['target']} action={entry['action']} success={entry['success']} measured={entry['measured_ms']:>6.2f}ms"
            )


def run_demo() -> None:
    random.seed(42)
    modes = [
        ("A", ["1", "2", "3", "4"]),
        ("B", ["1", "A", "2", "B"]),
    ]

    for mode, targets in modes:
        provider = TMTTaskProvider(mode=mode, targets=targets)
        agent = CognitiveAgent(provider)
        agent.run()
        print("-" * 92)


if __name__ == "__main__":
    run_demo()
