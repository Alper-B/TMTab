import csv
from pathlib import Path

ANALYSIS_RESULTS_CSV = Path(__file__).resolve().parent / "analysis_results.csv"
TASK_SWITCH_SUMMARY_CSV = Path(__file__).resolve().parent / "task_switching_summary.csv"
DEFAULT_DISTANCE_THRESHOLD = 60.0
DEFAULT_CLICK_TIME_THRESHOLD = 800.0


def load_analysis_results(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row["distance_threshold"] = float(row["distance_threshold"])
            row["click_time_threshold"] = float(row["click_time_threshold"])
            row["total_time_ms"] = float(row["total_time_ms"])
            row["average_click_time_ms"] = float(row["average_click_time_ms"])
            row["total_targets"] = int(row["total_targets"])
            row["completed_targets"] = int(row["completed_targets"])
            rows.append(row)
    return rows


def group_by_participant_task(rows):
    groups = {}
    for row in rows:
        key = (row["participant"], row["task_type"])
        groups.setdefault(key, []).append(row)
    return groups


def summarize_group(rows):
    if not rows:
        return None

    default_row = next(
        (
            row
            for row in rows
            if row["distance_threshold"] == DEFAULT_DISTANCE_THRESHOLD
            and row["click_time_threshold"] == DEFAULT_CLICK_TIME_THRESHOLD
        ),
        None,
    )

    average_click_times = [row["average_click_time_ms"] for row in rows]
    total_times = [row["total_time_ms"] for row in rows]

    best_avg_click = min(rows, key=lambda row: row["average_click_time_ms"])
    worst_avg_click = max(rows, key=lambda row: row["average_click_time_ms"])
    best_total_time = min(rows, key=lambda row: row["total_time_ms"])
    worst_total_time = max(rows, key=lambda row: row["total_time_ms"])

    return {
        "sample_count": len(rows),
        "default": default_row,
        "mean_average_click_time_ms": sum(average_click_times) / len(average_click_times),
        "mean_total_time_ms": sum(total_times) / len(total_times),
        "best_average_click": best_avg_click,
        "worst_average_click": worst_avg_click,
        "best_total_time": best_total_time,
        "worst_total_time": worst_total_time,
    }


def format_summary(participant, task, summary):
    if summary is None:
        return f"No data available for participant={participant}, task={task}\n"

    lines = [
        f"Participant {participant} | Task {task}",
        f"Rows analyzed: {summary['sample_count']}",
        f"Default thresholds: distance={DEFAULT_DISTANCE_THRESHOLD}, click_time={DEFAULT_CLICK_TIME_THRESHOLD}",
    ]

    if summary["default"] is not None:
        default = summary["default"]
        lines.extend(
            [
                f"Default average click time (ms): {default['average_click_time_ms']}",
                f"Default total task time (ms): {default['total_time_ms']}",
                f"Default completed targets: {default['completed_targets']} / {default['total_targets']}",
            ]
        )
    else:
        lines.append("Default-threshold row not found in analysis data.")

    lines.extend(
        [
            f"Mean average click time across thresholds (ms): {round(summary['mean_average_click_time_ms'], 3)}",
            f"Mean total time across thresholds (ms): {round(summary['mean_total_time_ms'], 3)}",
            "Best average click time across threshold combinations:",
            f"  {round(summary['best_average_click']['average_click_time_ms'], 3)} ms at distance={summary['best_average_click']['distance_threshold']} click_time={summary['best_average_click']['click_time_threshold']}",
            "Worst average click time across threshold combinations:",
            f"  {round(summary['worst_average_click']['average_click_time_ms'], 3)} ms at distance={summary['worst_average_click']['distance_threshold']} click_time={summary['worst_average_click']['click_time_threshold']}",
            "Best total task time across threshold combinations:",
            f"  {round(summary['best_total_time']['total_time_ms'], 3)} ms at distance={summary['best_total_time']['distance_threshold']} click_time={summary['best_total_time']['click_time_threshold']}",
            "Worst total task time across threshold combinations:",
            f"  {round(summary['worst_total_time']['total_time_ms'], 3)} ms at distance={summary['worst_total_time']['distance_threshold']} click_time={summary['worst_total_time']['click_time_threshold']}",
        ]
    )

    return "\n".join(lines) + "\n"


def print_all_summaries():
    if not ANALYSIS_RESULTS_CSV.exists():
        print(f"Analysis CSV not found: {ANALYSIS_RESULTS_CSV}")
        return

    rows = load_analysis_results(ANALYSIS_RESULTS_CSV)
    groups = group_by_participant_task(rows)

    for participant in ["subA", "subB", "subC"]:
        for task in ["A", "B"]:
            summary = summarize_group(groups.get((participant, task), []))
            print(format_summary(participant, task, summary))
            print("-" * 80)


def main():
    print_all_summaries()


if __name__ == "__main__":
    main()
