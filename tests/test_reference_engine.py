"""Independent, hand-calculated regression checks for the causal event engine.

Run from the repository root with ``python -m unittest discover -s tests -v``. Expected times below are calculated from the declared experiment
parameters; the tests do not call the simulator's prediction/service helpers to
construct their expected answers.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'engine/reference'))
import frozen_policy_engine as simulation


def task(task_id, arrival_ms=0.0, demand_mi=100.0, payload_kb=0.0,
         deadline_ms=10_000.0, propagation_ms=0.0):
    return {
        "task_id": task_id,
        "scenario": "hand_calculated",
        "run_id": 1,
        "seed": 123,
        "arrival_time_s": arrival_ms / 1000.0,
        "cpu_demand_mi": demand_mi,
        "task_size_kb": payload_kb,
        "deadline_ms": deadline_ms,
        "edge_cloud_network_delay_ms": propagation_ms,
    }


def parameters(**overrides):
    values = {
        "scenario": "hand_calculated",
        "edge_nodes": 1,
        "cloud_servers": 1,
        "edge_service_rate_tps_each": 10.0,
        "cloud_service_rate_tps_each": 100.0,
        "edge_queue_capacity": 5,
        "cloud_queue_capacity": 5,
        "network_model": "independent",
        "bandwidth_mbps": 25.0,
    }
    values.update(overrides)
    return values


class EventEngineTests(unittest.TestCase):
    def simulate(self, rows, mode="edge_only", **overrides):
        frame = pd.DataFrame(rows)
        original = frame.copy(deep=True)
        settings = parameters(**overrides)
        settings_before = dict(settings)
        detailed, summary = simulation.simulate_subset(frame, settings, mode)
        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(settings, settings_before, "Simulation changed its input parameters")
        self.assertEqual(len(detailed), len(rows))
        self.assertEqual(set(detailed["task_id"]), {row["task_id"] for row in rows})
        self.assertFalse(detailed["task_id"].duplicated().any())
        return detailed.set_index("task_id", drop=False), summary

    def assert_time(self, row, field, expected):
        self.assertAlmostEqual(float(row[field]), expected, places=3, msg=field)

    def test_idle_edge_executes_without_queueing(self):
        # At 10 nominal tasks/s, 100 MI takes 100 ms, after a 10 s idle period.
        detail, summary = self.simulate([task("idle", arrival_ms=10_000)])
        row = detail.loc["idle"]
        self.assertEqual(row["status"], "completed")
        self.assert_time(row, "start_time_ms", 10_000.0)
        self.assert_time(row, "finish_time_ms", 10_100.0)
        self.assert_time(row, "waiting_time_ms", 0.0)
        self.assert_time(row, "response_time_ms", 100.0)
        self.assertEqual(summary["blocked_tasks"], 0)
        self.assertEqual(summary["avg_edge_queue_length"], 0.0)

    def test_zero_waiting_buffer_accepts_idle_server_and_rejects_busy_server(self):
        detail, summary = self.simulate(
            [task("first"), task("busy", 1), task("release", 100)],
            edge_queue_capacity=0,
        )
        self.assertEqual(detail.loc["first", "status"], "completed")
        self.assertEqual(detail.loc["busy", "status"], "blocked")
        self.assertEqual(detail.loc["release", "status"], "completed")
        self.assert_time(detail.loc["first"], "finish_time_ms", 100.0)
        self.assert_time(detail.loc["release"], "start_time_ms", 100.0)
        self.assert_time(detail.loc["release"], "finish_time_ms", 200.0)
        self.assertEqual(summary["completed_tasks"], 2)
        self.assertEqual(summary["blocked_tasks"], 1)

    def test_every_policy_can_use_idle_servers_with_no_waiting_slots(self):
        for mode in simulation.PROCESSING_MODES:
            with self.subTest(mode=mode):
                detail, summary = self.simulate(
                    [task("idle")], mode=mode,
                    edge_queue_capacity=0, cloud_queue_capacity=0,
                )
                self.assertEqual(detail.loc["idle", "status"], "completed")
                self.assert_time(detail.loc["idle"], "waiting_time_ms", 0.0)
                self.assertEqual(summary["completed_tasks"], 1)
                self.assertEqual(summary["blocked_tasks"], 0)

    def test_finite_waiting_buffer_and_same_time_release(self):
        # A runs [0,100], B waits [0,100] then runs [100,200], C is blocked.
        # At t=100 B has left the waiting buffer, so D can wait [100,200].
        detail, summary = self.simulate(
            [task("A"), task("B"), task("C"), task("D", 100)],
            edge_queue_capacity=1,
        )
        self.assertEqual(detail.loc["C", "status"], "blocked")
        for name, start, finish, wait in [
            ("A", 0.0, 100.0, 0.0),
            ("B", 100.0, 200.0, 100.0),
            ("D", 200.0, 300.0, 100.0),
        ]:
            self.assertEqual(detail.loc[name, "status"], "completed")
            self.assert_time(detail.loc[name], "start_time_ms", start)
            self.assert_time(detail.loc[name], "finish_time_ms", finish)
            self.assert_time(detail.loc[name], "waiting_time_ms", wait)
        self.assertEqual(summary["completed_tasks"], 3)
        self.assertEqual(summary["blocked_tasks"], 1)
        self.assertAlmostEqual(summary["avg_edge_queue_length"], 2.0 / 3.0, places=8)
        self.assertAlmostEqual(summary["edge_utilization"], 1.0, places=8)

    def test_return_delay_does_not_keep_cloud_server_busy(self):
        # Cloud compute lasts 100 ms; the separate 5 ms return adds to response.
        # Consequently the next job may start at t=100, before A's result returns.
        detail, summary = self.simulate(
            [task("A"), task("busy", 1), task("B", 100)],
            mode="cloud_only", cloud_queue_capacity=0,
            cloud_service_rate_tps_each=10.0,
        )
        self.assertEqual(detail.loc["busy", "status"], "blocked")
        self.assert_time(detail.loc["A"], "start_time_ms", 0.0)
        self.assert_time(detail.loc["A"], "finish_time_ms", 105.0)
        self.assert_time(detail.loc["B"], "start_time_ms", 100.0)
        self.assert_time(detail.loc["B"], "finish_time_ms", 205.0)
        self.assertEqual(summary["completed_tasks"], 2)

    def test_independent_transfer_can_reverse_cloud_arrival_order(self):
        # A uploads 625 KB for 200 ms. B is submitted at t=10 with no payload.
        # B must therefore use the cloud [10,110], ahead of A [200,300].
        for mode in ("cloud_only", "hybrid_cloud_edge", "deadline_aware_hybrid"):
            with self.subTest(mode=mode):
                detail, _ = self.simulate(
                    [task("A", payload_kb=625.0), task("B", 10)],
                    mode=mode, edge_service_rate_tps_each=1.0,
                    cloud_service_rate_tps_each=10.0,
                )
                for name, start, finish in [("A", 200.0, 305.0), ("B", 10.0, 115.0)]:
                    self.assertEqual(detail.loc[name, "processing_location"], "cloud")
                    self.assert_time(detail.loc[name], "start_time_ms", start)
                    self.assert_time(detail.loc[name], "finish_time_ms", finish)
                    self.assert_time(detail.loc[name], "waiting_time_ms", 0.0)

    def test_current_cloud_congestion_changes_hybrid_decision(self):
        # A occupies cloud [0,100], B reserves [100,200]. At t=2, C needs
        # 100 ms at idle edge versus 198+10+5=213 ms in the congested cloud.
        for mode in ("hybrid_cloud_edge", "deadline_aware_hybrid"):
            with self.subTest(mode=mode):
                detail, _ = self.simulate(
                    [task("A", demand_mi=1000), task("B", 1, 1000), task("C", 2)],
                    mode=mode,
                )
                self.assertEqual(detail.loc["A", "processing_location"], "cloud")
                self.assertEqual(detail.loc["B", "processing_location"], "cloud")
                self.assert_time(detail.loc["B"], "start_time_ms", 100.0)
                self.assertEqual(detail.loc["C", "processing_location"], "edge")
                self.assert_time(detail.loc["C"], "start_time_ms", 2.0)
                self.assert_time(detail.loc["C"], "finish_time_ms", 102.0)
                self.assertEqual(detail.loc["C", "cloud_queue_at_decision"], 1)
                self.assert_time(detail.loc["C"], "predicted_cloud_wait_ms", 198.0)

    def test_cloud_arrival_precedes_source_decision_at_same_timestamp(self):
        # A has zero upload/propagation and immediately occupies cloud at t=0.
        # B's source event at the same instant must observe that occupancy.
        detail, _ = self.simulate(
            [task("A", demand_mi=1000), task("B")], mode="hybrid_cloud_edge"
        )
        self.assertEqual(detail.loc["A", "processing_location"], "cloud")
        self.assertEqual(detail.loc["B", "processing_location"], "edge")
        self.assert_time(detail.loc["B"], "predicted_cloud_wait_ms", 100.0)
        self.assert_time(detail.loc["B"], "finish_time_ms", 100.0)

    def test_future_cloud_forecast_does_not_purge_current_waiting_job(self):
        # C forecasts a cloud arrival after t=1000 but stays at edge. That
        # forecast must not remove B, which is still waiting at t=3.
        for mode in ("hybrid_cloud_edge", "deadline_aware_hybrid"):
            with self.subTest(mode=mode):
                detail, _ = self.simulate(
                    [task("A", demand_mi=1000), task("B", 1, 1000),
                     task("C", 2, payload_kb=3125), task("D", 3, 1000)],
                    mode=mode,
                )
                self.assertEqual(detail.loc["C", "processing_location"], "edge")
                self.assertEqual(detail.loc["D", "cloud_queue_at_decision"], 1)
                self.assertEqual(detail.loc["D", "processing_location"], "cloud")
                self.assert_time(detail.loc["D"], "start_time_ms", 200.0)
                self.assert_time(detail.loc["D"], "waiting_time_ms", 197.0)

    def test_forecasting_inflight_job_does_not_reserve_real_cloud_in_future(self):
        # A is known but in flight until t=100. B's long-horizon forecast may
        # replay A on a copy. C arrives at cloud at t=2 and must run immediately.
        for mode in ("hybrid_cloud_edge", "deadline_aware_hybrid"):
            with self.subTest(mode=mode):
                detail, _ = self.simulate(
                    [task("A", demand_mi=1000, payload_kb=312.5),
                     task("B", 1, payload_kb=3125), task("C", 2)],
                    mode=mode,
                )
                self.assertEqual(detail.loc["A", "processing_location"], "cloud")
                self.assertEqual(detail.loc["B", "processing_location"], "edge")
                self.assertEqual(detail.loc["C", "processing_location"], "cloud")
                self.assert_time(detail.loc["C"], "start_time_ms", 2.0)
                self.assert_time(detail.loc["C"], "finish_time_ms", 17.0)
                self.assert_time(detail.loc["A"], "start_time_ms", 100.0)

    def test_inflight_forecast_respects_cloud_waiting_capacity(self):
        # A arrives at cloud at t=100 and occupies its only server until 200.
        # B's upload would arrive at t=101; with no waiting slots that route is
        # infeasible. The deadline policy's declared capacity fallback uses edge.
        rows = [task("A", payload_kb=312.5), task("B", 1, payload_kb=312.5)]
        for mode in ("hybrid_cloud_edge", "deadline_aware_hybrid"):
            with self.subTest(mode=mode):
                detail, _ = self.simulate(
                    rows, mode=mode, edge_service_rate_tps_each=1.0,
                    cloud_service_rate_tps_each=10.0, cloud_queue_capacity=0,
                )
                self.assert_time(detail.loc["A"], "start_time_ms", 100.0)
                self.assertFalse(detail.loc["B", "predicted_cloud_available"])
                self.assert_time(detail.loc["B"], "predicted_cloud_wait_ms", 99.0)
                if mode == "deadline_aware_hybrid":
                    self.assertEqual(detail.loc["B", "processing_location"], "edge")
                    self.assert_time(detail.loc["B"], "finish_time_ms", 1001.0)
                else:
                    self.assertEqual(detail.loc["B", "status"], "blocked")
                    self.assertEqual(detail.loc["B", "failure_reason"], "cloud_queue_capacity")

    def test_utilization_baseline_keeps_threshold_equality_at_edge(self):
        # Four of five edge servers are busy at t=1: utilization=0.8, so E
        # still uses the edge. At t=2 all five are busy, so F offloads.
        detail, _ = self.simulate(
            [task("A"), task("B"), task("C"), task("D"),
             task("E", 1), task("F", 2)],
            mode="utilization_based", edge_nodes=5, utilization_threshold=0.8,
        )
        self.assertEqual(detail.loc["E", "processing_location"], "edge")
        self.assertAlmostEqual(detail.loc["E", "edge_utilization_at_decision"], 0.8)
        self.assert_time(detail.loc["E"], "start_time_ms", 1.0)
        self.assertEqual(detail.loc["F", "processing_location"], "cloud")
        self.assert_time(detail.loc["F"], "finish_time_ms", 17.0)

    def test_network_baseline_virtual_probe_and_threshold_boundary(self):
        # A's 4096 KB upload occupies the link for 1310.72 ms. A 1024 KB
        # virtual probe at t=0 takes 1310.72+327.68 ms, giving available
        # throughput 5 Mbps and congestion 1-5/25=0.8. Equality selects edge.
        # The probe is observational: it must add no actual upload traffic.
        detail, _ = self.simulate(
            [task("A", payload_kb=4096), task("B")],
            mode="network_aware", network_model="shared_fcfs",
            network_probe_kb=1024.0, network_utilization_threshold=0.8,
            network_baseline_rule="paper_utilization",
        )
        self.assertEqual(detail.loc["A", "processing_location"], "cloud")
        self.assert_time(detail.loc["A"], "start_time_ms", 1310.72)
        self.assert_time(detail.loc["A"], "finish_time_ms", 1325.72)
        self.assertAlmostEqual(detail.loc["B", "wan_available_bandwidth_estimate_mbps"], 5.0)
        self.assertAlmostEqual(detail.loc["B", "estimated_wan_utilization"], 0.8)
        self.assertEqual(detail.loc["B", "processing_location"], "edge")
        self.assert_time(detail.loc["B"], "finish_time_ms", 100.0)

    def test_load_baseline_counts_busy_and_waiting_jobs_per_server(self):
        # Edge has two servers, cloud one. The current normalized loads
        # (edge,cloud) before A..E are (0,0),(.5,0),(.5,1),(1,1),(1.5,1).
        detail, _ = self.simulate(
            [task("A"), task("B"), task("C"), task("D"), task("E", 1)],
            mode="queue_length_based", edge_nodes=2,
            cloud_service_rate_tps_each=10.0,
        )
        self.assertEqual(
            detail["processing_location"].tolist(),
            ["edge", "cloud", "edge", "edge", "cloud"],
        )
        self.assert_time(detail.loc["D"], "start_time_ms", 100.0)
        self.assert_time(detail.loc["E"], "start_time_ms", 100.0)

    def test_shared_fcfs_link_serializes_uploads(self):
        # Each 312.5 KB task consumes 100 ms on the shared 25 Mbps link.
        # Five ms compute and five ms return give final responses 110 and 210.
        detail, _ = self.simulate(
            [task("A", payload_kb=312.5), task("B", payload_kb=312.5)],
            mode="cloud_only", network_model="shared_fcfs", cloud_servers=2,
            cloud_service_rate_tps_each=1000.0,
        )
        self.assert_time(detail.loc["A"], "start_time_ms", 100.0)
        self.assert_time(detail.loc["B"], "start_time_ms", 200.0)
        self.assert_time(detail.loc["A"], "finish_time_ms", 110.0)
        self.assert_time(detail.loc["B"], "finish_time_ms", 210.0)
        self.assert_time(detail.loc["A"], "network_queue_wait_ms", 0.0)
        self.assert_time(detail.loc["B"], "network_queue_wait_ms", 100.0)

    def test_independent_link_has_no_shared_upload_queue(self):
        detail, _ = self.simulate(
            [task("A", payload_kb=312.5), task("B", payload_kb=312.5)],
            mode="cloud_only", network_model="independent", cloud_servers=2,
            cloud_service_rate_tps_each=1000.0,
        )
        for name in ("A", "B"):
            self.assert_time(detail.loc[name], "start_time_ms", 100.0)
            self.assert_time(detail.loc[name], "finish_time_ms", 110.0)
            self.assert_time(detail.loc[name], "network_queue_wait_ms", 0.0)

    def test_shared_bandwidth_change_changes_exact_upload_times(self):
        # At 50 Mbps, the same pair uploads in 50 ms each: responses 60/110.
        detail, _ = self.simulate(
            [task("A", payload_kb=312.5), task("B", payload_kb=312.5)],
            mode="cloud_only", network_model="shared_fcfs", bandwidth_mbps=50.0,
            cloud_servers=2, cloud_service_rate_tps_each=1000.0,
        )
        self.assert_time(detail.loc["A"], "finish_time_ms", 60.0)
        self.assert_time(detail.loc["B"], "finish_time_ms", 110.0)

    def test_deadline_equality_counts_as_success(self):
        detail, _ = self.simulate([
            task("equal", deadline_ms=100),
            task("late", 200, deadline_ms=99),
            task("early", 400, deadline_ms=101),
        ])
        self.assertEqual(detail.loc["equal", "sla_violation"], 0)
        self.assertEqual(detail.loc["late", "sla_violation"], 1)
        self.assertEqual(detail.loc["early", "sla_violation"], 0)
        self.assert_time(detail.loc["equal"], "deadline_slack_ms", 0.0)
        cloud, _ = self.simulate(
            [task("cloud_equal", deadline_ms=105)], mode="cloud_only",
            cloud_service_rate_tps_each=10.0,
        )
        self.assertEqual(cloud.loc["cloud_equal", "sla_violation"], 0)

    def test_configured_observation_horizon_bounds_utilization_and_throughput(self):
        # A 100 ms service within a declared 10 s observation consumes 1% of
        # one edge server's observation capacity, even though the job ends early.
        _, summary = self.simulate([task("A")], duration_s=10.0)
        self.assertAlmostEqual(summary["observation_duration_s"], 10.0)
        self.assertAlmostEqual(summary["drain_end_s"], 10.0)
        self.assertAlmostEqual(summary["edge_utilization"], 0.01)
        self.assertAlmostEqual(summary["throughput_tasks_per_s"], 0.1)
        self.assertAlmostEqual(summary["offered_rate_tps"], 0.1)

    def test_cloud_rejection_retains_consumed_upload_proxy(self):
        # A runs on cloud [0,1000]. B uploads 1024 KB, reaches a busy cloud at
        # 327.68 ms, and is rejected because the waiting capacity is zero.
        # Its completed upload still costs 0.00002 and 0.002 proxy units.
        detail, summary = self.simulate(
            [task("A", demand_mi=1000), task("B", payload_kb=1024)],
            mode="cloud_only", cloud_queue_capacity=0,
            cloud_service_rate_tps_each=10.0,
        )
        self.assertEqual(detail.loc["B", "status"], "blocked")
        self.assertAlmostEqual(detail.loc["B", "cost_proxy"], 0.00002)
        self.assertAlmostEqual(detail.loc["B", "energy_proxy"], 0.002)
        self.assertAlmostEqual(summary["total_cost_proxy"], 0.25002)
        self.assertAlmostEqual(summary["total_energy_proxy"], 18.002)
        self.assertAlmostEqual(summary["cost_proxy_per_offered_task"], 0.12501)
        self.assertAlmostEqual(summary["avg_cost_proxy"], 0.25002)
        self.assertAlmostEqual(summary["on_time_completion_rate"], 0.5)
        self.assertAlmostEqual(summary["unsuccessful_task_rate"], 0.5)

    def test_task_conservation_and_response_identity_for_every_policy(self):
        rows = [
            task("A", demand_mi=1000, payload_kb=125),
            task("B", demand_mi=100, payload_kb=25),
            task("C", 1, demand_mi=10, payload_kb=0),
            task("D", 2, demand_mi=1000, payload_kb=250),
            task("E", 3, demand_mi=100, deadline_ms=1),
            task("F", 1000, demand_mi=100, payload_kb=25),
        ]
        for mode in simulation.PROCESSING_MODES:
            for network in ("independent", "shared_fcfs"):
                with self.subTest(mode=mode, network=network):
                    detail, summary = self.simulate(
                        rows, mode=mode, network_model=network,
                        edge_queue_capacity=1, cloud_queue_capacity=1,
                    )
                    self.assertEqual(set(detail["status"]) - {"completed", "blocked"}, set())
                    completed = detail[detail["status"] == "completed"]
                    blocked = detail[detail["status"] == "blocked"]
                    self.assertEqual(summary["total_tasks"], len(rows))
                    self.assertEqual(summary["completed_tasks"], len(completed))
                    self.assertEqual(summary["blocked_tasks"], len(blocked))
                    self.assertEqual(len(completed) + len(blocked), len(rows))
                    self.assertGreaterEqual(summary["edge_utilization"], 0.0)
                    self.assertLessEqual(summary["edge_utilization"], 1.0 + 1e-9)
                    self.assertGreaterEqual(summary["cloud_utilization"], 0.0)
                    self.assertLessEqual(summary["cloud_utilization"], 1.0 + 1e-9)
                    for _, row in completed.iterrows():
                        self.assertGreaterEqual(row["waiting_time_ms"], 0.0)
                        self.assertGreaterEqual(row["response_time_ms"], row["service_time_ms"])
                        self.assertAlmostEqual(
                            row["response_time_ms"],
                            row["finish_time_ms"] - row["arrival_time_s"] * 1000.0,
                            places=3,
                        )
                    self.assertTrue(blocked["finish_time_ms"].isna().all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
