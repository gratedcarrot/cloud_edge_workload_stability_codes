"""Check observed trace timing, initialization filtering, and resource mappings."""
import csv
import gzip
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import build_public_trace_data as build

def event(timestamp, event_type=0, job=1, index=0):
    return [timestamp, "", job, index, "", event_type, "user", 1, 5, 0.1, 0.2, 0.01, 0]

def write_events(path: Path, rows: list[list]) -> None:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as stream:
        csv.writer(stream).writerows(rows)

def raw_frame(times):
    return pd.DataFrame([event(t, job=i + 1) for i, t in enumerate(times)], columns=build.TASK_EVENT_COLUMNS)

class TraceExtractionTests(unittest.TestCase):
    def test_initial_state_excluded_and_true_burst_gaps_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "part.csv.gz"
            write_events(path, [event(0), event(599_999_999), event(600_000_000, 1),
                                event(600_000_000, job=2), event(600_000_000, job=3),
                                event(600_100_000, job=4), event(601_000_000, job=5)])
            provenance = []
            raw = build.collect_submit_events([path], 4, provenance=provenance)
            self.assertEqual(raw["time"].tolist(), [600_000_000, 600_000_000, 600_100_000, 601_000_000])
            self.assertEqual(provenance[0]["excluded_before_observation_window"], 2)
            self.assertEqual(provenance[0]["excluded_non_submit"], 1)
            mapped = build.map_to_cloud_edge_format(raw)
            np.testing.assert_allclose(mapped["arrival_time_s"], [0, 0, 0.1, 1])
            np.testing.assert_allclose(mapped["inter_arrival_time_s"], [0, 0, 0.1, 0.9])
            self.assertEqual(mapped["lambda_at_arrival_tps"].tolist(), [3, 3, 3, 1])

    def test_partition_without_valid_arrivals_does_not_create_fake_events(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.csv", Path(directory) / "b.csv"
            write_events(a, [event(0), event(0)])
            write_events(b, [event(600_000_000), event(600_000_001, job=2)])
            reports = []
            raw = build.collect_submit_events([a, b], 2, provenance=reports)
            self.assertEqual(len(raw), 2)
            self.assertEqual(reports[0]["selected_records"], 0)
            self.assertTrue(reports[0]["complete_source_read"])
            self.assertEqual(raw["source_partition"].tolist(), ["b.csv", "b.csv"])

    def test_row_and_download_limits_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            write_events(path, [event(0), event(0), event(600_000_000), event(601_000_000)])
            with self.assertRaisesRegex(build.ExtractionLimitError, "row limit"):
                build.collect_submit_events([path], 2, budget=build.ExtractionBudget(max_rows=1))
            with self.assertRaisesRegex(build.ExtractionLimitError, "source-byte limit"):
                build.collect_submit_events([path], 2, budget=build.ExtractionBudget(max_source_bytes=5))

    def test_insufficient_records_do_not_silently_shorten_requested_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            write_events(path, [event(0), event(600_000_000)])
            with self.assertRaisesRegex(RuntimeError, "Only 1 eligible"):
                build.collect_submit_events([path], 2)

    def test_mapping_rejects_zero_initialization_and_degenerate_real_window(self):
        with self.assertRaisesRegex(ValueError, "Initial-state"):
            build.map_to_cloud_edge_format(raw_frame([0, 0]))
        with self.assertRaisesRegex(ValueError, "Degenerate"):
            build.map_to_cloud_edge_format(raw_frame([600_000_000, 600_000_000]))
        with self.assertRaises(ValueError):
            build.map_to_cloud_edge_format(raw_frame([600_000_000, np.nan]))