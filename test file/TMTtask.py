import json
import math
import sys
from datetime import datetime
from pathlib import Path

# Analyzer configuration constants
DEFAULT_LOG_FILE = "logs/participant_A_20260806_171028.jsonl"
DISTANCE_THRESHOLD = 60.0
CLICK_TIME_THRESHOLD = 800.0

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
    def __init__(self, dist_threshold=DISTANCE_THRESHOLD, click_time_threshold=CLICK_TIME_THRESHOLD):
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
                        misses.append(
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

        if last_click_time is None:
            total_time_ms = (end_time - start_time).total_seconds() * 1000.0
        else:
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
            "distance_threshold": self.dist_threshold,
            "click_time_threshold": self.click_time_threshold,
        }


def format_summary(summary):
    lines = [
        f"Task: {summary['task_type']}",
        f"Participant: {summary['participant_id']}",
        f"Total targets: {summary['total_targets']}",
        f"Completed targets: {summary['completed_targets']}",
        f"Total time (ms): {summary['total_time_ms']}",
        f"Average click time (ms): {summary['average_click_time_ms']}",
        f"Number of misses: {summary['miss_count']}",
        f"Distance threshold: {summary['distance_threshold']}",
        f"Click time threshold: {summary['click_time_threshold']}",
    ]
    if summary["miss_count"] > 0:
        lines.append("\nMiss details:")
        for idx, miss in enumerate(summary["misses"], start=1):
            lines.append(
                f"  {idx}. target={miss['target']} type={miss['type']} delay_ms={miss.get('delay_ms', miss.get('hover_duration_ms'))}"
            )
    return "\n".join(lines)


def main():
    logfile = Path(DEFAULT_LOG_FILE)
    if not logfile.exists():
        print(f"Default log file not found: {logfile}")
        return

    events = load_jsonl(logfile)
    analyzer = TMTLogAnalyzer()
    summary = analyzer.analyze(events)

    print(format_summary(summary))


if __name__ == "__main__":
    main()
