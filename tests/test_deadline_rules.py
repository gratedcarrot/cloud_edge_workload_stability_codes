"""Hand-constructed checks of deadline override, proxy tie and guarded routing branches."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'engine/reference'))
import frozen_policy_engine as new


def stage(components=None, **changes):
    values = dict(selected='cloud', reason='hybrid_queue_and_response_rule',
                  eavailable=True, cavailable=True, eresponse=500.0,
                  cresponse=1000.0, deadline=100.0, eservice=100.0,
                  cservice=40.0, payload_kb=0.0, params={}, ewait=400.0,
                  threshold_ms=220.0, components=new.deadline_component_settings(components))
    values.update(changes)
    return new.apply_deadline_stages(**values)


class AblationTests(unittest.TestCase):
    def test_least_lateness_bypasses_wait_gate_when_both_routes_miss(self):
        route, reason, log = stage()
        self.assertEqual((route, reason), ('edge', 'least_predicted_lateness'))
        self.assertTrue(log['least_lateness_bypassed_edge_wait_gate'])
        route, reason, log = stage({'least_lateness': False})
        self.assertEqual((route, reason), ('cloud', 'hybrid_queue_and_response_rule'))
        self.assertFalse(log['least_lateness_applied'])

    def test_removing_feasibility_override_leaves_one_feasible_route_branch_untouched(self):
        route, _, log = stage(deadline=600.0)
        self.assertEqual(route, 'edge')
        self.assertTrue(log['feasibility_override_changed_destination'])
        route, _, log = stage({'feasibility_override': False}, deadline=600.0)
        self.assertEqual(route, 'cloud')
        self.assertFalse(log['least_lateness_applied'])

    def test_proxy_tie_only_changes_close_and_slack_feasible_decision(self):
        settings = dict(eresponse=100.0, cresponse=90.0, deadline=500.0,
                        eservice=100.0, cservice=90.0, ewait=0.0)
        self.assertEqual(stage(**settings)[0], 'edge')
        self.assertEqual(stage({'proxy_tie_break': False}, **settings)[0], 'cloud')
        self.assertTrue(stage(**settings)[2]['proxy_tie_break_changed_destination'])

    def test_capacity_stage_can_be_removed_without_enabling_other_stages(self):
        settings = dict(cavailable=False)
        self.assertEqual(stage({'least_lateness': False}, **settings)[0], 'edge')
        self.assertEqual(stage({'capacity_fallback': False, 'least_lateness': False}, **settings)[0], 'cloud')

    def test_equal_lateness_leaves_route_unchanged(self):
        route, _, log = stage(eresponse=1000.0, cresponse=1000.0)
        self.assertEqual(route, 'cloud')
        self.assertFalse(log['least_lateness_applied'])

    def test_invalid_switches_rejected(self):
        for settings in ({'unknown': False}, {'least_lateness': 0}, {'proxy_tie_break': 'False'}):
            with self.assertRaises(ValueError):
                new.deadline_component_settings(settings)


if __name__ == '__main__':
    unittest.main()
