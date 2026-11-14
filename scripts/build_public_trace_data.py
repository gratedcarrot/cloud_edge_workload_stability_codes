from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_RECORDS = 20000
# Google documents the 2011 observation window as starting at 600 seconds.
# Time-zero events describe initial state and are not observed new arrivals.
OBSERVATION_START_US = 600_000_000
SOURCE_DOCUMENTATION = "https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md"
BASE_URL = "https://commondatastorage.googleapis.com/clusterdata-2011-2/task_events"
PART_FILES = [
    "part-00000-of-00500.csv.gz",
    "part-00001-of-00500.csv.gz",
    "part-00002-of-00500.csv.gz",
    "part-00003-of-00500.csv.gz",
    "part-00004-of-00500.csv.gz",
]

DOWNLOAD_DIR = Path("trace_downloads")
OUTPUT_DIR = Path("public_trace_data")

TASK_EVENT_COLUMNS = [
    "time",
    "missing_info",
    "job_id",
    "task_index",
    "machine_id",
    "event_type",
    "user",
    "scheduling_class",
    "priority",
    "resource_request_cpu",
    "resource_request_memory",
    "resource_request_disk",
    "different_machine_constraint",
]


class ExtractionLimitError(RuntimeError):
    """The explicitly bounded extraction could not collect its requested subset."""


@dataclass
class ExtractionBudget:
    max_source_bytes: int = 128 * 1024 * 1024
    max_uncompressed_bytes: int = 1024 * 1024 * 1024
    max_rows: int = 5_000_000
    max_seconds: float = 180.0
    source_bytes: int = 0
    uncompressed_bytes: int = 0
    rows: int = 0
    started: float = 0.0

    def __post_init__(self) -> None:
        if (min(self.max_source_bytes, self.max_uncompressed_bytes, self.max_rows, self.max_seconds) <= 0
                or not math.isfinite(self.max_seconds)):
            raise ValueError("Extraction limits must be positive.")
        self.started = time.monotonic()

    def check(self) -> None:
        if time.monotonic() - self.started > self.max_seconds:
            raise ExtractionLimitError("Extraction exceeded its wall-time limit.")
        if self.source_bytes > self.max_source_bytes:
            raise ExtractionLimitError("Extraction exceeded its source-byte limit.")
        if self.uncompressed_bytes > self.max_uncompressed_bytes:
            raise ExtractionLimitError("Extraction exceeded its uncompressed-byte limit.")
        if self.rows > self.max_rows:
            raise ExtractionLimitError("Extraction exceeded its scanned-row limit.")


class _CountedSource:
    """Hash exactly the source prefix read; this is not a full-shard checksum."""

    def __init__(self, handle, budget: ExtractionBudget) -> None:
        self.handle = handle
        self.budget = budget
        self.sha256 = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        self.budget.check()
        remaining = self.budget.max_source_bytes - self.budget.source_bytes
        # One extra byte permits an explicit cap error instead of fake EOF.
        size = min(size if size >= 0 else 65536, 65536, remaining + 1)
        value = self.handle.read(size)
        self.bytes_read += len(value)
        self.budget.source_bytes += len(value)
        self.sha256.update(value)
        self.budget.check()
        return value


def _read_source(source: str | Path, remaining: int, start_us: int,
                 budget: ExtractionBudget) -> tuple[pd.DataFrame, dict]:
    source_name = str(source)
    remote = source_name.startswith(("https://", "http://"))
    stats = {"source": source_name, "selected_records": 0, "scanned_rows": 0,
             "excluded_before_observation_window": 0, "excluded_non_submit": 0,
             "excluded_malformed": 0, "complete_source_read": False}
    rows = []
    handle = (urllib.request.urlopen(source_name, timeout=min(30, budget.max_seconds))
              if remote else Path(source).open("rb"))
    with handle:
        counted = _CountedSource(handle, budget)
        compressed = source_name.split("?", 1)[0].endswith(".gz")
        stream = gzip.GzipFile(fileobj=counted, mode="rb") if compressed else None

        def lines():
            if stream is not None:
                while True:
                    line = stream.readline()
                    if not line:
                        break
                    budget.uncompressed_bytes += len(line)
                    budget.check()
                    yield line.decode("utf-8")
            else:
                # Local uncompressed input is useful for small reproducible fixtures.
                pending = b""
                while True:
                    block = counted.read(65536)
                    if not block:
                        if pending:
                            budget.uncompressed_bytes += len(pending)
                            budget.check()
                            yield pending.decode("utf-8")
                        break
                    pending += block
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        budget.uncompressed_bytes += len(line) + 1
                        budget.check()
                        yield (line + b"\n").decode("utf-8")

        try:
            for record in csv.reader(lines()):
                budget.rows += 1
                stats["scanned_rows"] += 1
                budget.check()
                if len(record) != len(TASK_EVENT_COLUMNS):
                    stats["excluded_malformed"] += 1
                    continue
                try:
                    timestamp, event = int(record[0]), int(record[5])
                except ValueError:
                    stats["excluded_malformed"] += 1
                    continue
                if timestamp < start_us:
                    stats["excluded_before_observation_window"] += 1
                    continue
                if event != 0:
                    stats["excluded_non_submit"] += 1
                    continue
                rows.append(record)
                if len(rows) >= remaining:
                    break
            else:
                stats["complete_source_read"] = True
        finally:
            if stream is not None:
                stream.close()
            stats["source_bytes_read"] = counted.bytes_read
            stats["source_prefix_sha256"] = counted.sha256.hexdigest()
    stats["selected_records"] = len(rows)
    stats["checksum_scope"] = "Bytes consumed from source; partial when complete_source_read is false"
    frame = pd.DataFrame(rows, columns=TASK_EVENT_COLUMNS)
    frame["source_partition"] = source_name.rsplit("/", 1)[-1]
    return frame, stats


def read_submit_events_from_part(path: Path, remaining: int,
                                 start_us: int = OBSERVATION_START_US) -> pd.DataFrame:
    frame, _ = _read_source(path, remaining, start_us, ExtractionBudget())
    return frame


def collect_submit_events(paths: list[Path | str], target_records: int,
                          start_us: int = OBSERVATION_START_US,
                          budget: ExtractionBudget | None = None,
                          provenance: list[dict] | None = None) -> pd.DataFrame:
    if target_records < 2 or start_us < OBSERVATION_START_US:
        raise ValueError("Request at least two events at or after the 600-second observation start.")
    budget = budget or ExtractionBudget()
    collected, total = [], 0
    for path in paths:
        if total >= target_records:
            break
        part_df, stats = _read_source(path, target_records - total, start_us, budget)
        if provenance is not None:
            provenance.append(stats)
        if not part_df.empty:
            collected.append(part_df)
            total += len(part_df)
        print(f"Scanned {stats['scanned_rows']:,} rows from {path}; selected {len(part_df):,}")
    if total < target_records:
        raise RuntimeError(f"Only {total} eligible events found; requested {target_records}. "
                           "Provide more partitions or explicitly reduce --target-records.")
    result = pd.concat(collected, ignore_index=True)
    result["time"] = pd.to_numeric(result["time"], errors="raise").astype("int64")
    return result.sort_values("time", kind="stable", ignore_index=True)


def clean_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    return values.fillna(default)


def map_to_cloud_edge_format(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    for field in ("time", "job_id", "task_index"):
        df[field] = pd.to_numeric(df[field], errors="raise")
        if (not np.isfinite(df[field]).all() or (df[field] < 0).any()
                or (df[field] % 1 != 0).any()):
            raise ValueError(f"{field} must contain finite nonnegative integer source values.")
        df[field] = df[field].astype("int64")
    if (df["time"] < OBSERVATION_START_US).any():
        raise ValueError("Initial-state/pre-window events are not observed task arrivals; "
                         "select submit events at or after 600,000,000 microseconds.")
    df["priority"] = clean_numeric(df["priority"], 0)
    df["scheduling_class"] = clean_numeric(df["scheduling_class"], 0)
    df["resource_request_cpu"] = clean_numeric(df["resource_request_cpu"], 0)
    df["resource_request_memory"] = clean_numeric(df["resource_request_memory"], 0)
    df["resource_request_disk"] = clean_numeric(df["resource_request_disk"], 0)

    df = df[
        (df["resource_request_cpu"] >= 0)
        & (df["resource_request_memory"] >= 0)
        & (df["time"] >= 0)
    ].copy()

    if df.empty:
        raise RuntimeError("No valid records remained after cleaning the public trace data.")

    df = df.sort_values("time", kind="stable", ignore_index=True)

    first_time = float(df["time"].iloc[0])
    df["arrival_time_s"] = (df["time"] - first_time) / 1_000_000.0
    df["inter_arrival_time_s"] = df["arrival_time_s"].diff().fillna(df["arrival_time_s"])

    if df["arrival_time_s"].max() <= 0:
        raise ValueError("Degenerate trace: selected events have no positive time span. "
                         "Select a longer real observation window; artificial spacing is disabled.")

    df["task_type"] = (
        df["scheduling_class"]
        .astype(int)
        .map(
            {
                0: "background_task",
                1: "batch_job",
                2: "analytics_task",
                3: "service_task",
            }
        )
        .fillna("compute_task")
    )

    priority_bucket = pd.cut(
        df["priority"].clip(lower=0, upper=100),
        bins=[-1, 1, 4, 8, 100],
        labels=[1, 2, 3, 4],
    )
    df["priority_mapped"] = priority_bucket.astype(float).fillna(1).astype(int)

    cpu = df["resource_request_cpu"].clip(lower=0)
    mem = df["resource_request_memory"].clip(lower=0)
    disk = df["resource_request_disk"].clip(lower=0)

    df["cpu_demand_mi"] = (50 + cpu * 600 + mem * 200).round(3).clip(lower=1)
    df["memory_demand_mb"] = (64 + mem * 4096).round(3).clip(lower=8)
    df["task_size_kb"] = (64 + cpu * 2048 + mem * 1024 + disk * 512).round(3).clip(lower=1)

    df["deadline_ms"] = (
        250
        + (4 - df["priority_mapped"]) * 150
        + (3 - df["scheduling_class"].clip(0, 3)) * 100
    ).round().astype(int).clip(lower=100)

    zones = np.array(["edge_zone_A", "edge_zone_B", "edge_zone_C", "edge_zone_D"])
    zone_idx = (df["job_id"].astype(int) % 4).to_numpy()
    df["source_zone"] = zones[zone_idx]
    df["edge_node_candidate"] = ["edge_" + str(i + 1) for i in zone_idx]

    size_reference = float(df["task_size_kb"].quantile(0.95))
    if not np.isfinite(size_reference) or size_reference <= 0:
        size_reference = max(float(df["task_size_kb"].max()), 1.0)

    df["edge_cloud_network_delay_ms"] = (
        45 + (df["task_size_kb"] / size_reference).clip(0, 1) * 25
    ).round(3)

    out = pd.DataFrame(
        {
            "task_id": [
                f"GCT_{int(j)}_{int(t)}_{idx:05d}"
                for idx, (j, t) in enumerate(zip(df["job_id"], df["task_index"]))
            ],
            "scenario": "public_trace_validation",
            "run_id": 1,
            "arrival_time_s": df["arrival_time_s"].round(6),
            "inter_arrival_time_s": df["inter_arrival_time_s"].round(6),
            "lambda_at_arrival_tps": 0,
            "burst_flag": 0,
            "source_zone": df["source_zone"],
            "edge_node_candidate": df["edge_node_candidate"],
            "task_type": df["task_type"],
            "priority": df["priority_mapped"],
            "task_size_kb": df["task_size_kb"],
            "cpu_demand_mi": df["cpu_demand_mi"],
            "memory_demand_mb": df["memory_demand_mb"],
            "deadline_ms": df["deadline_ms"],
            "edge_cloud_network_delay_ms": df["edge_cloud_network_delay_ms"],
        }
    )

    out["arrival_second"] = np.floor(out["arrival_time_s"]).astype(int)
    counts = out.groupby("arrival_second")["task_id"].transform("count")
    out["lambda_at_arrival_tps"] = counts.astype(int)
    out.drop(columns=["arrival_second"], inplace=True)

    return out


def write_scenario_parameters(output_dir: Path) -> None:
    params = pd.DataFrame(
        [
            {
                "scenario": "public_trace_validation",
                "description": "Google ClusterData 2011 task_events-derived validation workload",
                "duration_s": 0.0,
                "base_lambda_tps": 0.0,
                "burst_start_s": None,
                "burst_end_s": None,
                "burst_lambda_tps": None,
                "edge_nodes": 4,
                "edge_service_rate_tps_each": 7,
                "cloud_servers": 3,
                "cloud_service_rate_tps_each": 18,
                "edge_queue_capacity": 120,
                "cloud_queue_capacity": 250,
                "mean_network_delay_ms": 55,
                "std_network_delay_ms": 0,
                "total_edge_capacity_tps": 28,
                "total_cloud_capacity_tps": 54,
                "total_system_capacity_tps": 82,
            }
        ]
    )
    params.to_csv(output_dir / "scenario_parameters.csv", index=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_summary_and_metadata(mapped: pd.DataFrame, output_dir: Path,
                               raw: pd.DataFrame | None = None,
                               extraction: dict | None = None,
                               pilot: bool = False) -> None:
    duration = float(mapped["arrival_time_s"].max() - mapped["arrival_time_s"].min())
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("A public-trace summary requires a finite positive observed time span.")

    summary = pd.DataFrame(
        [
            {
                "scenario": "public_trace_validation",
                "records": len(mapped),
                "duration_s": round(duration, 3),
                "observed_arrival_rate_tps": round(len(mapped) / duration, 3),
                "avg_task_size_kb": round(mapped["task_size_kb"].mean(), 3),
                "avg_cpu_demand_mi": round(mapped["cpu_demand_mi"].mean(), 3),
                "avg_memory_demand_mb": round(mapped["memory_demand_mb"].mean(), 3),
                "avg_deadline_ms": round(mapped["deadline_ms"].mean(), 3),
                "avg_network_delay_ms": round(mapped["edge_cloud_network_delay_ms"].mean(), 3),
                "time_normalized": False,
            }
        ]
    )
    summary.to_csv(output_dir / "public_trace_summary.csv", index=False)

    metadata = {
        "dataset_name": "Google datacenter trace-derived cloud-edge simulation input",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_role": "pipeline_pilot_only" if pilot else "simulation_input",
        "source_dataset": "Google ClusterData 2011-2 task_events",
        "source_url": "https://commondatastorage.googleapis.com/clusterdata-2011-2/",
        "source_documentation": SOURCE_DOCUMENTATION,
        "selected_records": int(len(mapped)),
        "selection_method": "First requested submit events (event_type=0) in supplied partition order, "
                            "at/after documented observation start; stable timestamp ordering",
        "documented_observation_start_us": OBSERVATION_START_US,
        "record_limit_may_truncate_final_timestamp_batch": True,
        "time_normalization": {"applied": False},
        "arrival_span_s": duration,
        "distinct_arrival_timestamps": int(mapped["arrival_time_s"].nunique()),
        "simultaneous_arrival_gaps": int((mapped["inter_arrival_time_s"].iloc[1:] == 0).sum()),
        "extraction": extraction or {},
        "mapping": {
            "observed_source_fields": ["time", "job_id", "task_index", "priority", "scheduling_class",
                                       "resource_request_cpu", "resource_request_memory", "resource_request_disk"],
            "arrival_time_s": "(observed timestamp_us - first retained timestamp_us) / 1e6; no jitter or invented spacing",
            "cpu_demand_mi": "50 + 600*CPU_request + 200*memory_request; assumed conversion, not measured instructions",
            "memory_demand_mb": "64 + 4096*memory_request; assumed conversion and metadata, not an enforced RAM admission limit",
            "task_size_kb": "64 + 2048*CPU_request + 1024*memory_request + 512*disk_request; assumed transfer size",
            "priority": "Original priority clipped to [0,100], bucketed using cutpoints [-1,1,4,8,100] into [1,2,3,4]",
            "deadline_ms": "250 + 150*(4-priority_bucket) + 100*(3-clip(scheduling_class,0,3)); synthetic deadline",
            "task_type": "Descriptive analyst labels inferred from scheduling_class; not measured application identity",
            "source_zone_and_edge_candidate": "job_id modulo 4; synthetic labels, not measured location or mobility",
            "edge_cloud_network_delay_ms": "45 + 25*clip(task_size_kb / p95(task_size_kb),0,1); modeled delay, not measured network performance",
            "lambda_at_arrival_tps": "Empirical task count in each one-second bin; not a measured instantaneous Poisson intensity",
            "burst_flag": "Constant zero placeholder; actual timestamp ties and gaps are retained independently",
            "missing_resource_priority_or_class_values": "Non-numeric/nonfinite values imputed as zero; negative CPU/memory rows excluded",
            "disk_request": "Negative values clipped to zero by the mapping",
            "capacity_network_energy_cost": "Chosen simulation assumptions; not inferred from the Google trace",
        },
        "sampling_limitations": ["One selected prefix is not a representative sample of all deployment conditions",
                                 "Repeated evaluation of this same input does not create independent trace samples",
                                 "Last timestamp group may be partially retained at the requested record count"],
        "mapping_note": (
            "Only submission timing and source resource requests/identifiers are observed. "
            "Task computation, transfers, deadlines, placement, edge/cloud capacities, cost and energy "
            "are modeled assumptions. This is not validation on a contemporary edge deployment."
        ),
    }
    if raw is not None:
        times = pd.to_numeric(raw["time"], errors="raise")
        metadata["source_selection"] = {"rows_selected": len(raw), "rows_after_mapping": len(mapped),
                                       "rows_removed_by_mapping": len(raw) - len(mapped),
                                       "first_timestamp_us": int(times.min()),
                                       "last_timestamp_us": int(times.max())}
        metadata["imputed_field_counts"] = {}
        for field in ("priority", "scheduling_class", "resource_request_cpu",
                      "resource_request_memory", "resource_request_disk"):
            values = pd.to_numeric(raw[field], errors="coerce")
            metadata["imputed_field_counts"][field] = int((~np.isfinite(values)).sum())
    for filename, key in (("selected_raw_submit_events.csv", "selected_raw_events_sha256"),
                          ("cloud_edge_workload_trace.csv", "raw_mapped_trace_sha256")):
        path = output_dir / filename
        if path.exists():
            metadata[key] = sha256_file(path)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")



def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build bounded, auditable Google datacenter trace-derived inputs.")
    parser.add_argument("--target-records", type=int, default=TARGET_RECORDS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--input", nargs="+", help="Local .csv/.csv.gz files or public URLs in partition order")
    parser.add_argument("--start-us", type=int, default=OBSERVATION_START_US)
    parser.add_argument("--max-source-mb", type=float, default=128)
    parser.add_argument("--max-uncompressed-mb", type=float, default=1024)
    parser.add_argument("--max-rows", type=int, default=5_000_000)
    parser.add_argument("--max-seconds", type=float, default=180)
    parser.add_argument("--pilot", action="store_true", help="Label output as pipeline pilot only")
    args = parser.parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = args.input or [str(DOWNLOAD_DIR / name) if (DOWNLOAD_DIR / name).is_file()
                           else f"{BASE_URL}/{name}" for name in PART_FILES]
    budget = ExtractionBudget(max_source_bytes=int(args.max_source_mb * 1024**2),
                              max_uncompressed_bytes=int(args.max_uncompressed_mb * 1024**2),
                              max_rows=args.max_rows, max_seconds=args.max_seconds)
    sources: list[dict] = []
    try:
        raw = collect_submit_events(paths, args.target_records, args.start_us, budget, sources)
    except Exception as exc:
        failure = {"error": str(exc), "requested_records": args.target_records,
                   "source_inputs": [str(path) for path in paths], "start_us": args.start_us,
                   "source_bytes_read": budget.source_bytes, "scanned_rows": budget.rows,
                   "elapsed_s": time.monotonic() - budget.started, "completed_source_reports": sources}
        (output_dir / "extraction_failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
        raise
    extraction = {"sources": sources, "requested_records": args.target_records, "start_us": args.start_us,
                  "source_bytes_read": budget.source_bytes, "uncompressed_bytes_processed": budget.uncompressed_bytes,
                  "scanned_rows": budget.rows, "elapsed_s": round(time.monotonic() - budget.started, 3),
                  "limits": {"max_source_bytes": budget.max_source_bytes,
                             "max_uncompressed_bytes": budget.max_uncompressed_bytes,
                             "max_rows": budget.max_rows, "max_seconds": budget.max_seconds}}
    # Keep extraction evidence even if a later mapping validity check rejects the subset.
    (output_dir / "extraction_provenance.json").write_text(json.dumps(extraction, indent=2), encoding="utf-8")
    raw.to_csv(output_dir / "selected_raw_submit_events.csv", index=False)

    print("Mapping public trace records to cloud-edge simulator format...")
    mapped = map_to_cloud_edge_format(raw)

    mapped.to_csv(output_dir / "cloud_edge_workload_trace.csv", index=False)
    write_scenario_parameters(output_dir)
    params_path = output_dir / "scenario_parameters.csv"
    params = pd.read_csv(params_path)
    span = float(mapped["arrival_time_s"].max())
    params.loc[0, "duration_s"] = span
    params.loc[0, "base_lambda_tps"] = len(mapped) / span
    params.to_csv(params_path, index=False)
    write_summary_and_metadata(mapped, output_dir, raw, extraction, args.pilot)

    print()
    print("Public trace-derived validation dataset created successfully.")
    print(f"Output folder: {output_dir.resolve()}")
    print(f"Records: {len(mapped):,}")
    print()
    print("Summary:")
    print(pd.read_csv(output_dir / "public_trace_summary.csv").to_string(index=False))


if __name__ == "__main__":
    main()
