import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Analyzer configuration constants
PARTICIPANTS_DIR = Path(__file__).resolve().parents[1] / "particpants"
ANALYSIS_OUTPUT_CSV = Path(__file__).resolve().parent / "analysis_results.csv"
TASK_SWITCH_OUTPUT_CSV = Path(__file__).resolve().parent / "task_switching_summary.csv"
PLOTS_DIR = Path(__file__).resolve().parent / "analysis_plots"

# Threshold sweeps for batch analysis (10 values each)
DISTANCE_THRESHOLDS = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 120.0]
CLICK_TIME_THRESHOLDS = [200.0, 400.0, 600.0, 800.0, 1000.0, 1200.0, 1400.0, 1600.0, 1800.0, 2000.0]

DEFAULT_DISTANCE_THRESHOLD = 60.0
DEFAULT_CLICK_TIME_THRESHOLD = 800.0

CANVAS_SIZE = 780
CANVAS_MARGIN = 30
CANVAS_ACTIVE = CANVAS_SIZE - 2 * CANVAS_MARGIN


def load_jsonl(path):
    events = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def normalized_to_canvas(coord):
    return (coord / 100.0) * CANVAS_ACTIVE + CANVAS_MARGIN


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def parse_timestamp(ts):
    if ts is None:
        return None
    return datetime.fromisoformat(ts)


class TMTLogAnalyzer:
    def __init__(self, dist_threshold=DEFAULT_DISTANCE_THRESHOLD, click_time_threshold=DEFAULT_CLICK_TIME_THRESHOLD):
        self.dist_threshold = dist_threshold
        self.click_time_threshold = click_time_threshold

    def analyze(self, events):
        if not events:
            raise ValueError("Log file contains no events")

        header = next((e for e in events if e.get("event_type") == "task_started"), events[0])
        task_type = header.get("task_type", "--")
        targets = header.get("targets", [])
        layout = header.get("layout", [])
        target_positions = {
            target: (
                normalized_to_canvas(pos[0]),
                normalized_to_canvas(pos[1]),
            )
            for target, pos in zip(targets, layout)
        }

        start_time = None
        end_time = None
        first_click_time = None
        last_click_time = None
        completed_target_count = 0
        misses = []
        late_clicks = []
        current_hover = None
        hover_start = None
        hover_target = None
        hover_reported = False

        for event in events:
            ts = parse_timestamp(event.get("timestamp"))
            if ts is None:
                continue
            if start_time is None:
                start_time = ts
            end_time = ts

            event_type = event.get("event_type")
            if event_type in {"correct_click", "incorrect_click", "miss_click"}:
                if first_click_time is None:
                    first_click_time = ts
                last_click_time = ts
                completed_target_count = event.get("completed_count", completed_target_count)

                clicked_target = event.get("target")
                if current_hover is not None and hover_target == clicked_target and not hover_reported:
                    click_delay = (ts - hover_start).total_seconds() * 1000.0
                    if click_delay > self.click_time_threshold:
                        late_clicks.append(
                            {
                                "target": hover_target,
                                "type": "late_click",
                                "hover_start_ms": hover_start.isoformat(timespec="milliseconds"),
                                "click_time_ms": ts.isoformat(timespec="milliseconds"),
                                "delay_ms": round(click_delay, 3),
                            }
                        )
                        hover_reported = True
                    else:
                        hover_reported = True
                current_hover = None
                hover_start = None
                hover_target = None
                hover_reported = False

            if event_type == "mouse_move":
                current_target = event.get("current_target")
                if current_target and current_target in target_positions:
                    x = event.get("x")
                    y = event.get("y")
                    pos = (x, y)
                    center = target_positions[current_target]
                    if distance(pos, center) <= self.dist_threshold:
                        current_hover = current_target
                        if hover_target != current_target:
                            hover_target = current_target
                            hover_start = ts
                            hover_reported = False
                        elif hover_start is not None and not hover_reported:
                            elapsed = (ts - hover_start).total_seconds() * 1000.0
                            if elapsed >= self.click_time_threshold:
                                misses.append(
                                    {
                                        "target": hover_target,
                                        "type": "hover_no_click",
                                        "hover_start_ms": hover_start.isoformat(timespec="milliseconds"),
                                        "flagged_at_ms": ts.isoformat(timespec="milliseconds"),
                                        "hover_duration_ms": round(elapsed, 3),
                                    }
                                )
                                hover_reported = True
                    else:
                        if current_target != hover_target:
                            current_hover = None
                            hover_start = None
                            hover_target = None
                            hover_reported = False

        total_time_ms = (end_time - start_time).total_seconds() * 1000.0
        average_click_time_ms = total_time_ms / max(len(targets), 1)

        return {
            "task_type": task_type,
            "participant_id": header.get("participant_id"),
            "total_time_ms": round(total_time_ms, 3),
            "average_click_time_ms": round(average_click_time_ms, 3),
            "total_targets": len(targets),
            "completed_targets": completed_target_count,
            "miss_count": len(misses),
            "misses": misses,
            "late_click_count": len(late_clicks),
            "late_clicks": late_clicks,
            "distance_threshold": self.dist_threshold,
            "click_time_threshold": self.click_time_threshold,
        }


def get_participant_log_files(participants_dir):
    participant_files = {}
    for subdir in sorted(participants_dir.iterdir()):
        if not subdir.is_dir():
            continue
        task_files = {}
        for path in sorted(subdir.glob("*.jsonl")):
            name = path.name.lower()
            if "participant_a" in name and "tmt" not in name:
                task_files["A"] = path
            elif "participant_b" in name and "tmt" not in name:
                task_files["B"] = path
            elif "_a_" in name:
                task_files["A"] = path
            elif "_b_" in name:
                task_files["B"] = path
        if task_files:
            participant_files[subdir.name] = task_files
    return participant_files


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_miss_surface(participant, task, distance_thresholds, click_time_thresholds, miss_grid, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        miss_grid,
        cmap="viridis",
        origin="lower",
        aspect="auto",
        extent=[min(distance_thresholds), max(distance_thresholds), min(click_time_thresholds), max(click_time_thresholds)],
    )
    fig.colorbar(im, ax=ax, label="Miss count")
    ax.set_xlabel("Distance threshold")
    ax.set_ylabel("Click time threshold (ms)")
    ax.set_title(f"{participant} {task} miss counts")
    ax.set_xticks(distance_thresholds)
    ax.set_yticks(click_time_thresholds)
    ax.set_xticklabels([str(int(d)) for d in distance_thresholds], rotation=45)
    ax.set_yticklabels([str(int(c)) for c in click_time_thresholds])
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def batch_analyze(participant_logs, distance_thresholds, click_time_thresholds):
    rows = []
    miss_grids = {}
    default_summaries = {}

    for participant, tasks in sorted(participant_logs.items()):
        miss_grids[participant] = {}
        for task, path in sorted(tasks.items()):
            events = load_jsonl(path)
            grid = np.zeros((len(click_time_thresholds), len(distance_thresholds)), dtype=int)
            for i_dist, dist in enumerate(distance_thresholds):
                for j_click, click_time in enumerate(click_time_thresholds):
                    analyzer = TMTLogAnalyzer(dist, click_time)
                    summary = analyzer.analyze(events)
                    rows.append(
                        {
                            "participant": participant,
                            "task_type": task,
                            "log_file": str(path),
                            "distance_threshold": dist,
                            "click_time_threshold": click_time,
                            "total_time_ms": summary["total_time_ms"],
                            "average_click_time_ms": summary["average_click_time_ms"],
                            "total_targets": summary["total_targets"],
                            "completed_targets": summary["completed_targets"],
                            "miss_count": summary["miss_count"],
                        }
                    )
                    grid[j_click, i_dist] = summary["miss_count"]

            miss_grids[participant][task] = grid
            default_analyzer = TMTLogAnalyzer(DEFAULT_DISTANCE_THRESHOLD, DEFAULT_CLICK_TIME_THRESHOLD)
            default_summaries[(participant, task)] = default_analyzer.analyze(events)

    task_switch_rows = []
    for participant in sorted(participant_logs):
        a_summary = default_summaries.get((participant, "A"))
        b_summary = default_summaries.get((participant, "B"))
        if a_summary is None or b_summary is None:
            continue
        task_switch_rows.append(
            {
                "participant": participant,
                "tmt_a_average_click_time_ms": a_summary["average_click_time_ms"],
                "tmt_b_average_click_time_ms": b_summary["average_click_time_ms"],
                "switch_time_ms": round(a_summary["average_click_time_ms"] - b_summary["average_click_time_ms"], 3),
                "distance_threshold": DEFAULT_DISTANCE_THRESHOLD,
                "click_time_threshold": DEFAULT_CLICK_TIME_THRESHOLD,
            }
        )

    return rows, task_switch_rows, miss_grids


def main():
    participants_dir = PARTICIPANTS_DIR
    if not participants_dir.exists():
        print(f"Participants directory not found: {participants_dir}")
        return

    participant_logs = get_participant_log_files(participants_dir)
    if not participant_logs:
        print(f"No participant logs found under: {participants_dir}")
        return

    rows, switch_rows, miss_grids = batch_analyze(participant_logs, DISTANCE_THRESHOLDS, CLICK_TIME_THRESHOLDS)

    fieldnames = [
        "participant",
        "task_type",
        "log_file",
        "distance_threshold",
        "click_time_threshold",
        "total_time_ms",
        "average_click_time_ms",
        "total_targets",
        "completed_targets",
        "miss_count",
    ]
    write_csv(ANALYSIS_OUTPUT_CSV, fieldnames, rows)
    write_csv(TASK_SWITCH_OUTPUT_CSV, [
        "participant",
        "tmt_a_average_click_time_ms",
        "tmt_b_average_click_time_ms",
        "switch_time_ms",
        "distance_threshold",
        "click_time_threshold",
    ], switch_rows)

    for participant, tasks in miss_grids.items():
        for task, grid in tasks.items():
            plot_path = PLOTS_DIR / f"{participant}_{task}_miss_surface.png"
            plot_miss_surface(participant, task, DISTANCE_THRESHOLDS, CLICK_TIME_THRESHOLDS, grid, plot_path)

    print(f"Batch analysis complete.")
    print(f"Results CSV: {ANALYSIS_OUTPUT_CSV}")
    print(f"Task-switch summary CSV: {TASK_SWITCH_OUTPUT_CSV}")
    print(f"Plots directory: {PLOTS_DIR}")
    print(f"Participants analyzed: {', '.join(sorted(participant_logs))}")
    print(f"Threshold combinations: {len(DISTANCE_THRESHOLDS)} x {len(CLICK_TIME_THRESHOLDS)} = {len(rows)} rows")


if __name__ == "__main__":
    main()
