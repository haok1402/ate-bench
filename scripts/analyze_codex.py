#!/usr/bin/env python3
"""
analyze_codex.py <snapshot>

Read the codex session transcripts under <snapshot> and report the Table 5 metrics
for the question-and-answer suite. Definitions mirror the paper's Claude backend
(Pith-Train-Analysis/sources/analyze_agent_efforts.py); we read the equivalent
quantities out of codex `token_count` events and record timestamps instead of
Anthropic usage blocks.

For each question and framework we report, over the independent attempts, the
median of four per-attempt quantities:

    Session Duration  active wall-clock minutes (idle gaps capped, see below)
    Agent Turns #     number of model turns (one codex `token_count` event each)
    Per-Turn Context  p50 of the per-turn context (input_tokens) within the run
    Output Tokens     total output tokens the agent generated

Per-Turn Context is a MEDIAN OF MEDIANS: within one attempt we take the p50 of the
per-turn context sizes, then across attempts we take the median of those p50s --
matching analyze_agent_efforts.py, which reports `percentile(ctx_per_turn, 50)`
per attempt (EFFORT_ROWS) and `_median` across attempts (_bolded_row).

Session Duration mirrors compute_agent_active_intervals + compute_wall_clock_seconds:
between consecutive timestamped records, an interval is counted in full when the
gap is <= 600s or when it spans a tool call (a `function_call` whose `call_id` is
answered by the next record's `function_call_output`); otherwise only the first
600s is credited, so a long idle wait does not inflate the metric. Intervals are
then merged and summed. This is the agent's own active time; GPU jobs are not part of it.

Active GPU Time is a separate metric reported for the training categories
(new-features, operate-and-profile). It accumulates the durations of the run's
artifacts/{train,verify}-*.log segments, each running from the 'YYYYMMDD-HHMMSS' stamp
in its filename to the last 'exit-time:' line inside it. Logs missing that marker are
skipped, so a single absent exit-time stays small in impact. The Q&A suite is read-only
and produces no such logs. Session Duration and Active GPU Time are reported in minutes.

Column sets follow the paper: the Q&A median table omits Session Duration (Table 5), while
the Q&A per-attempt tables carry it and spell out the question titles (Tables 9 and 10).

A codex `token_count` event carries `input_tokens` that already includes the
cached prefix (`cached_input_tokens` is a subset), so it is the direct analog of
the Claude backend's `input + cache_read + cache_creation` per-turn context.

<snapshot> is any directory; every run beneath it (a directory holding a
sessions/*.jsonl transcript) is discovered automatically, so it works on a single
run, one question, or the whole snapshots tree.
"""

import argparse
import json
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

from utilities import attempt_records_table, effort_table, format_k

FRAMEWORKS = [
    ("Megatron-LM", "Megatron-LM"),
    ("torchtitan", "TorchTitan"),
    ("pith-train", "PithTrain"),
]

IDLE_THRESHOLD_SEC = 600


def detect_framework(run_name: str):
    """Map a run directory name (e.g. 'pith-train-80e86f') to a framework key."""
    for key, *_ in FRAMEWORKS:
        if run_name == key or run_name.startswith(key + "-"):
            return key
    return None


def percentile(xs, p):
    """
    p-th percentile with the paper backend's nearest-rank convention
    (analyze_agent_efforts.py: index = min(n - 1, int(p/100 * n))).
    """
    if not xs:
        return 0
    s = sorted(xs)
    i = min(len(s) - 1, int(p / 100.0 * len(s)))
    return s[i]


def parse_iso(s):
    """Parse an ISO-8601 timestamp (optional trailing Z) to tz-aware UTC, or None."""
    if not isinstance(s, str):
        return None
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def iter_records(session_file: Path):
    """Yield each JSON record from a JSONL transcript, skipping blank/malformed lines."""
    with session_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                continue


def read_usage(session_file: Path):
    """
    Per-turn context sizes and the run's total output tokens from one transcript.

    Returns (per_turn_input, total_output) where per_turn_input has one entry per
    model turn (each codex `token_count` event's last_token_usage.input_tokens),
    or None if the transcript has no usage records.
    """
    per_turn_input = []
    total_output = 0
    for record in iter_records(session_file):
        if record.get("type") != "event_msg":
            continue
        payload = record.get("payload") or {}
        if payload.get("type") != "token_count":
            continue
        info = payload.get("info") or {}
        last = info.get("last_token_usage") or {}
        per_turn_input.append(last.get("input_tokens") or 0)
        total = info.get("total_token_usage") or {}
        if total:
            total_output = total.get("output_tokens") or total_output
    if not per_turn_input:
        return None
    return per_turn_input, total_output


def active_seconds(transcripts, idle_threshold_sec=IDLE_THRESHOLD_SEC):
    """
    Active wall-clock seconds across a run's transcripts (idle gaps capped).

    Consecutive timestamped records form an interval; it is counted in full when
    the gap is <= idle_threshold_sec or when it spans a tool call (a function_call
    answered by the next record's function_call_output), otherwise only the first
    idle_threshold_sec is credited. Intervals are then merged and summed.
    """
    events = []
    for transcript in transcripts:
        for record in iter_records(transcript):
            ts = parse_iso(record.get("timestamp"))
            if ts is None:
                continue
            payload = record.get("payload") or {}
            initiated, completed = set(), set()
            call_id = payload.get("call_id")
            if call_id and payload.get("type") == "function_call":
                initiated.add(call_id)
            elif call_id and payload.get("type") == "function_call_output":
                completed.add(call_id)
            events.append((ts, initiated, completed))
    events.sort(key=lambda e: e[0])
    if len(events) < 2:
        return 0.0

    threshold = timedelta(seconds=idle_threshold_sec)
    intervals = []
    for i in range(1, len(events)):
        prev_ts, prev_init, _ = events[i - 1]
        cur_ts, _, cur_comp = events[i]
        gap = cur_ts - prev_ts
        spans_tool = bool(prev_init & cur_comp)
        if spans_tool or gap <= threshold:
            intervals.append((prev_ts, cur_ts))
        else:
            intervals.append((prev_ts, prev_ts + threshold))

    intervals.sort()
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum((end - start).total_seconds() for start, end in merged)


def parse_stamp(s):
    """
    Parse a 'YYYYMMDD-HHMMSS' stamp (log filename or exit-time) to a UTC-aware datetime, or None.

    The stamp carries no offset; UTC is attached so the value is tz-aware. Only differences
    between two stamps from the same run are ever taken, so the choice of zone cancels out.
    """
    try:
        return datetime.strptime(s, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def last_exit_time(log_file: Path):
    """The datetime of the last 'exit-time: <stamp>' line in a log, or None if it has none."""
    stamp = None
    with log_file.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("exit-time:"):
                stamp = line.split(":", 1)[1].strip()
    return parse_stamp(stamp) if stamp else None


def active_gpu_seconds(run_dir: Path):
    """
    Total GPU wall-clock seconds for a run, accumulated over its artifacts/{train,verify}-*.log.

    Each log is one training segment: it starts at the 'YYYYMMDD-HHMMSS' stamp in its filename
    and ends at the last 'exit-time:' line inside it. Logs missing that marker are skipped;
    segments are summed, so a single missing exit-time stays small in impact (the report takes
    the median across attempts). The Q&A suite is read-only and produces no such logs.
    """
    artifacts = Path(run_dir, "artifacts")
    if not artifacts.is_dir():
        return 0.0
    total = 0.0
    for log in sorted(artifacts.glob("*.log")):
        if not (log.name.startswith("train-") or log.name.startswith("verify-")):
            continue
        start = parse_stamp(log.stem.split("-", 1)[1])
        end = last_exit_time(log)
        if start is None or end is None:
            continue
        seconds = (end - start).total_seconds()
        if seconds > 0:
            total += seconds
    return total


def discover_runs(root: Path):
    """Yield (category, challenge, framework, metrics) for every codex run under root."""
    for sessions in sorted(root.rglob("sessions")):
        if not sessions.is_dir():
            continue
        transcripts = sorted(sessions.glob("*.jsonl"))
        if not transcripts:
            continue
        run = sessions.parent
        if not Path(run, "codex-launch.sh").is_file():
            continue
        framework = detect_framework(run.name)
        if framework is None:
            continue
        challenge = run.parent.name
        category = run.parent.parent.name
        per_turn_input = []
        total_output = 0
        for transcript in transcripts:
            series = read_usage(transcript)
            if series is None:
                continue
            per_turn_input.extend(series[0])
            total_output += series[1]
        if not per_turn_input:
            continue
        metrics = {
            "duration_min": active_seconds(transcripts) / 60.0,
            "active_gpu_min": active_gpu_seconds(run) / 60.0,
            "agent_turns": len(per_turn_input),
            "per_turn_context": percentile(per_turn_input, 50),
            "output_tokens": total_output,
        }
        yield category, challenge, framework, metrics


METRICS = {
    "duration_min": ("Session Duration", "min"),
    "active_gpu_min": ("Active GPU Time", "min"),
    "agent_turns": ("Agent Turns", "int"),
    "per_turn_context": ("Per-Turn Context", "k"),
    "output_tokens": ("Output Tokens", "k"),
}

QA_FIELDS = ["agent_turns", "per_turn_context", "output_tokens"]
QA_ATTEMPT_FIELDS = ["duration_min", "agent_turns", "per_turn_context", "output_tokens"]
OP_FIELDS = ["duration_min", "active_gpu_min", "agent_turns", "per_turn_context", "output_tokens"]

QA = "question-and-answer"
CATEGORY_ORDER = ["question-and-answer", "operate-and-profile", "new-features"]
CATEGORY_TITLE = {
    "question-and-answer": "Question & Answer",
    "operate-and-profile": "Operate & Profile",
    "new-features": "New Features",
}

QA_QUESTION_TITLES = {}
QA_QUESTION_TITLES["process-groups-device-mesh"] = "Process Groups / Device Mesh"
QA_QUESTION_TITLES["configuration-propagation"] = "Configuration Propagation"
QA_QUESTION_TITLES["data-loading-sharding"] = "Data Loading & Sharding"
QA_QUESTION_TITLES["distributed-seed-management"] = "Distributed Seed Management"
QA_QUESTION_TITLES["attention-kernel-dispatch"] = "Attention Kernel Dispatch"
QA_QUESTION_TITLES["rope-implementation"] = "RoPE Implementation"
QA_QUESTION_TITLES["swiglu-mlp-block"] = "SwiGLU / MLP Block"
QA_QUESTION_TITLES["normalization-placement"] = "Normalization Placement"
QA_QUESTION_TITLES["context-sequence-parallelism"] = "Context / Sequence Parallelism"
QA_QUESTION_TITLES["fsdp-ddp-wrapping"] = "FSDP / DDP Wrapping"
QA_QUESTION_TITLES["global-gradient-clipping"] = "Global Gradient Clipping"
QA_QUESTION_TITLES["distributed-checkpoint-serialization"] = "Distributed Checkpoint Serialization"

QA_QUESTION_ORDER = list(QA_QUESTION_TITLES)

CATEGORY_TASK_ORDER = {}
CATEGORY_TASK_ORDER["operate-and-profile"] = [
    "getting-started",
    "train-and-evaluate",
    "collect-routing-trace",
    "report-heavy-kernels",
]
CATEGORY_TASK_ORDER["new-features"] = [
    "differential-transformer",
    "dynamic-mixture-of-experts",
    "mixture-of-block-attention",
    "moe-plus-plus",
]


def format_cell(field, value):
    """Format a metric value per its field kind: minutes, integer, or K-suffixed."""
    kind = METRICS[field][1]
    if kind == "min":
        return f"{value:.1f}"
    if kind == "int":
        return str(int(value)) if float(value).is_integer() else f"{value:.1f}"
    return format_k(value)


def titleize(name):
    """Turn a kebab-case challenge name into a Title Case label."""
    return name.replace("-", " ").title()


def question_label(challenge, with_title):
    """
    Label for one Q&A challenge: 'Q5' for the median table (paper Table 5), or
    'Q5: Attention Kernel Dispatch' for the per-attempt tables (paper Tables 9 and 10).
    A challenge missing from QA_QUESTION_TITLES falls back to its title-cased name.
    """
    if challenge not in QA_QUESTION_ORDER:
        return titleize(challenge)
    number = f"Q{QA_QUESTION_ORDER.index(challenge) + 1}"
    return f"{number}: {QA_QUESTION_TITLES[challenge]}" if with_title else number


def category_layout(category, present, per_attempt=False):
    """
    Shared per-category table layout: metric columns, item ordering, header label, and
    whether the category renders horizontally (Q&A) or vertically. `present` is the set
    of challenge names seen for the category. `per_attempt` selects the appendix layout,
    which for Q&A carries Session Duration and spells out the question titles, matching
    the paper's Tables 9 and 10. Returns (metrics, items, item_header, horizontal).
    """
    horizontal = category == QA
    if horizontal:
        fields = QA_ATTEMPT_FIELDS if per_attempt else QA_FIELDS
    else:
        fields = OP_FIELDS
    metrics = [(f, METRICS[f][0]) for f in fields]
    if horizontal:
        challenges = [ch for ch in QA_QUESTION_ORDER if ch in present]
        challenges += sorted(present - set(QA_QUESTION_ORDER))
        items = [(ch, question_label(ch, per_attempt)) for ch in challenges]
        return metrics, items, "Task" if per_attempt else "#", horizontal
    order = CATEGORY_TASK_ORDER.get(category, [])
    challenges = [ch for ch in order if ch in present]
    challenges += sorted(present - set(order))
    items = [(ch, titleize(ch)) for ch in challenges]
    return metrics, items, "Task", horizontal


def main():
    """Discover codex runs, build the per-category tables, and write the markdown report."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "snapshot", type=Path, help="directory to scan for codex sessions"
    )
    parser.add_argument(
        "--category", help="restrict to one challenge category (default: all present)"
    )
    args = parser.parse_args()
    if not args.snapshot.is_dir():
        parser.error(f"{args.snapshot} is not a directory")

    groups: dict[tuple[str, str, str], list[dict]] = {}
    for category, challenge, framework, metrics in discover_runs(args.snapshot):
        if args.category and category != args.category:
            continue
        groups.setdefault((category, challenge, framework), []).append(metrics)
    if not groups:
        raise SystemExit(
            f"no codex runs with usage records found under {args.snapshot}"
        )

    def median_of(category, challenge, framework, field):
        """Median across attempts (median of per-attempt p50s for per_turn_context)."""
        attempts = groups.get((category, challenge, framework))
        if not attempts:
            return None
        return statistics.median(a[field] for a in attempts)

    blocks = ["# Aggregate Records"]

    categories = {cat for cat, _, _ in groups}
    ordered = [c for c in CATEGORY_ORDER if c in categories] + sorted(
        categories - set(CATEGORY_ORDER)
    )

    for category in ordered:
        present = {ch for cat, ch, _ in groups if cat == category}
        metrics, items, item_header, horizontal = category_layout(category, present)

        def raw_of(item_key, fw_key, metric_key, _cat=category):
            """Median value for one cell, or None if that framework has no attempts."""
            return median_of(_cat, item_key, fw_key, metric_key)

        blocks.append(f"## {CATEGORY_TITLE.get(category, titleize(category))}")
        blocks.append(
            effort_table(
                items,
                FRAMEWORKS,
                metrics,
                raw_of,
                format_cell,
                orient="horizontal" if horizontal else "vertical",
                item_header=item_header,
            )
        )

        name_of = dict(items)
        partial = [
            f"{name_of[ch]}/{label} (n={len(groups.get((category, ch, key), []))})"
            for ch, _ in items
            for key, label in FRAMEWORKS
            if len(groups.get((category, ch, key), [])) != 3
        ]
        if partial:
            blocks.append(f"_Attempts not equal to 3: {', '.join(partial)}._")

    blocks.append("# Per-Attempt Records")

    for category in ordered:
        present = {ch for cat, ch, _ in groups if cat == category}
        metrics, items, item_header, _ = category_layout(category, present, per_attempt=True)

        def attempts_of(item_key, fw_key, _cat=category):
            """Per-attempt metric dicts for one item x framework, in discovery order."""
            return groups.get((_cat, item_key, fw_key), [])

        blocks.append(f"## {CATEGORY_TITLE.get(category, titleize(category))}")
        blocks.append(
            attempt_records_table(
                items,
                FRAMEWORKS,
                metrics,
                attempts_of,
                format_cell,
                item_header=item_header,
            )
        )

    report = "\n\n".join(blocks) + "\n"
    output = Path(args.snapshot, "results", "codex.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
