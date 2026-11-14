"""Causal event simulation for the manuscript's finite-queue routing model.

Task demand and the task's nominal propagation delay are available at arrival.
Only tasks already submitted can influence forecasts. A forecast copies queue
state and replays already-dispatched uploads; it never advances live state.
"""
from __future__ import annotations

import hashlib
import heapq
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROCESSING_MODES = [
    'edge_only', 'cloud_only', 'hybrid_cloud_edge',
    'deadline_aware_hybrid', 'utilization_based', 'network_aware',
    'queue_length_based',
]
HYBRID_EDGE_DELAY_THRESHOLD_MS = 220.0
MIN_SERVICE_TIME_MS = 5.0
NOMINAL_CPU_DEMAND_MI = 100.0
EDGE_CLOUD_BANDWIDTH_MBPS = 25.0
RESULT_RETURN_DELAY_FACTOR = 0.65
MIN_RETURN_DELAY_MS = 5.0
EDGE_COST_PER_SERVICE_MS = 0.00004
CLOUD_COST_PER_SERVICE_MS = 0.00025
NETWORK_COST_PER_MB = 0.00002
EDGE_ENERGY_PER_SERVICE_MS = 0.004
CLOUD_ENERGY_PER_SERVICE_MS = 0.018
NETWORK_ENERGY_PER_MB = 0.002


@dataclass
class LayerState:
    server_available_times: List[float]
    waiting_start_heap: List[float] = field(default_factory=list)
    queue_events: List[Tuple[float, int]] = field(default_factory=list)
    total_busy_time_ms: float = 0.0


def service_time_ms(cpu_demand_mi: float, service_rate_tps_each: float) -> float:
    if cpu_demand_mi <= 0 or service_rate_tps_each <= 0:
        raise ValueError('Task demand and service rate must be positive.')
    return max(MIN_SERVICE_TIME_MS,
               1000.0 / float(service_rate_tps_each)
               * max(float(cpu_demand_mi), 1.0) / NOMINAL_CPU_DEMAND_MI)


def transfer_delay_ms(task_size_kb: float, bandwidth_mbps: float) -> float:
    if task_size_kb < 0 or bandwidth_mbps <= 0:
        raise ValueError('Payload must be nonnegative and bandwidth positive.')
    return float(task_size_kb) * 8.0 / float(bandwidth_mbps)


def return_delay_ms(network_delay_ms: float) -> float:
    return max(MIN_RETURN_DELAY_MS, float(network_delay_ms) * RESULT_RETURN_DELAY_FACTOR)


def make_empty_layer_state(server_count: int) -> LayerState:
    if int(server_count) < 1:
        raise ValueError('Each configured compute tier needs at least one server.')
    return LayerState([0.0] * int(server_count))


def purge_started_waiting_tasks(state: LayerState, time_ms: float) -> None:
    """Mutate only at the simulator's current time, never at a forecast time."""
    while state.waiting_start_heap and state.waiting_start_heap[0] <= time_ms:
        heapq.heappop(state.waiting_start_heap)


def waiting_queue_length(state: LayerState, time_ms: float) -> int:
    """Read-only query, including when called with a future forecast time."""
    return sum(start > time_ms for start in state.waiting_start_heap)


def choose_earliest_server(server_available_times: List[float]) -> int:
    return min(range(len(server_available_times)), key=server_available_times.__getitem__)


def estimate_waiting_delay_ms(state: LayerState, arrival_time_ms: float) -> float:
    return max(0.0, min(state.server_available_times) - float(arrival_time_ms))


def busy_servers(state: LayerState, time_ms: float) -> int:
    return sum(end > time_ms for end in state.server_available_times)


def can_accept(state: LayerState, time_ms: float, queue_capacity: int) -> bool:
    return (min(state.server_available_times) <= time_ms
            or waiting_queue_length(state, time_ms) < queue_capacity)


def add_queue_event(state: LayerState, time_ms: float, delta: int) -> None:
    state.queue_events.append((float(time_ms), int(delta)))


def schedule_on_layer(state: LayerState, arrival_to_layer_ms: float,
                      service_ms: float, queue_capacity: int) -> Dict:
    if queue_capacity < 0:
        raise ValueError('Waiting-buffer capacity cannot be negative.')
    purge_started_waiting_tasks(state, arrival_to_layer_ms)
    q = len(state.waiting_start_heap)
    server = choose_earliest_server(state.server_available_times)
    start = max(float(arrival_to_layer_ms), state.server_available_times[server])
    wait = start - float(arrival_to_layer_ms)
    if wait > 0 and q >= queue_capacity:
        return dict(accepted=False, start_time_ms=None, service_finish_time_ms=None,
                    waiting_time_ms=None, queue_length_at_arrival=q,
                    queue_length_after_assignment=q)
    finish = start + float(service_ms)
    state.server_available_times[server] = finish
    state.total_busy_time_ms += float(service_ms)
    if wait > 0:
        heapq.heappush(state.waiting_start_heap, start)
        add_queue_event(state, arrival_to_layer_ms, 1)
        add_queue_event(state, start, -1)
    return dict(accepted=True, start_time_ms=start, service_finish_time_ms=finish,
                waiting_time_ms=wait, queue_length_at_arrival=q,
                queue_length_after_assignment=len(state.waiting_start_heap))


def compute_average_queue_length(queue_events, simulation_end_ms: float) -> float:
    if simulation_end_ms <= 0:
        return 0.0
    current, previous, area = 0, 0.0, 0.0
    for when, delta in sorted(queue_events, key=lambda item: item[0]):
        when = min(float(when), simulation_end_ms)
        area += current * max(0.0, when - previous)
        current += delta
        previous = when
    area += current * max(0.0, simulation_end_ms - previous)
    return float(area / simulation_end_ms)


def task_size_mb(task_size_kb: float) -> float:
    # Retains the submitted proxy's KB-to-MB accounting convention.
    return max(0.0, float(task_size_kb) / 1024.0)


def cost_energy_proxy(processing_location: str, service_time_ms: Optional[float],
                      task_size_kb: float, params: Optional[Dict] = None) -> Tuple[float, float]:
    """Normalized accounting proxies; not prices, power, or measured energy."""
    p = params or {}
    service = max(0.0, float(service_time_ms or 0.0))
    if processing_location == 'edge':
        return (p.get('edge_cost_per_service_ms', EDGE_COST_PER_SERVICE_MS) * service,
                p.get('edge_energy_per_service_ms', EDGE_ENERGY_PER_SERVICE_MS) * service)
    if processing_location == 'cloud':
        size = task_size_mb(task_size_kb)
        return (p.get('cloud_cost_per_service_ms', CLOUD_COST_PER_SERVICE_MS) * service
                + p.get('network_cost_per_mb', NETWORK_COST_PER_MB) * size,
                p.get('cloud_energy_per_service_ms', CLOUD_ENERGY_PER_SERVICE_MS) * service
                + p.get('network_energy_per_mb', NETWORK_ENERGY_PER_MB) * size)
    return 0.0, 0.0


@dataclass
class Uplink:
    bandwidth_mbps: float = EDGE_CLOUD_BANDWIDTH_MBPS
    model: str = 'shared_fcfs'
    next_free_ms: float = 0.0
    total_busy_ms: float = 0.0

    def predict(self, now_ms: float, size_kb: float) -> Tuple[float, float]:
        wait = max(0.0, self.next_free_ms - now_ms) if self.model == 'shared_fcfs' else 0.0
        return wait, transfer_delay_ms(size_kb, self.bandwidth_mbps)

    def submit(self, now_ms: float, size_kb: float) -> Tuple[float, float, float]:
        wait, service = self.predict(now_ms, size_kb)
        end = now_ms + wait + service
        if self.model == 'shared_fcfs':
            self.next_free_ms = end
        self.total_busy_ms += service
        return wait, service, end

    def probe(self, now_ms: float, probe_kb: float) -> Tuple[float, float]:
        """Virtual probe: no extra packet or future arrival is inserted.

        Effective available throughput = probe bits / predicted upload sojourn.
        Congestion indicator = 1 - available throughput / configured capacity.
        This is a model-specific adaptation of the paper's network observation.
        """
        wait, service = self.predict(now_ms, probe_kb)
        available = probe_kb * 8.0 / (wait + service)
        utilization = max(0.0, min(1.0, 1.0 - available / self.bandwidth_mbps))
        return float(available), float(utilization)


def forecast_cloud(state: LayerState, inflight: Dict, arrival_ms: float,
                   queue_capacity: int) -> Tuple[float, int, bool]:
    """Forecast using known admissions and previously dispatched uploads only."""
    projected = LayerState(list(state.server_available_times),
                           list(state.waiting_start_heap))
    for item in sorted(inflight.values(), key=lambda x: (x['arrival_ms'], x['order'])):
        if item['arrival_ms'] > arrival_ms:
            break
        schedule_on_layer(projected, item['arrival_ms'], item['service_ms'], queue_capacity)
    return (estimate_waiting_delay_ms(projected, arrival_ms),
            waiting_queue_length(projected, arrival_ms),
            can_accept(projected, arrival_ms, queue_capacity))


def workload_fingerprint(subset: pd.DataFrame) -> str:
    cols = ['task_id', 'arrival_time_s', 'task_size_kb', 'cpu_demand_mi',
            'deadline_ms', 'edge_cloud_network_delay_ms']
    encoded = subset[cols].to_csv(index=False, float_format='%.17g').encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _validate(subset: pd.DataFrame, params: Dict, mode: str) -> None:
    if mode not in PROCESSING_MODES:
        raise ValueError(f'Unknown policy: {mode}')
    if subset.empty:
        raise ValueError('A run must contain at least one offered task.')
    if subset['task_id'].duplicated().any():
        raise ValueError('Task identifiers must be unique within a run.')
    if subset['scenario'].nunique() != 1 or subset['run_id'].nunique() != 1:
        raise ValueError('simulate_subset requires exactly one scenario/run.')
    for col in ['arrival_time_s', 'task_size_kb', 'cpu_demand_mi',
                'deadline_ms', 'edge_cloud_network_delay_ms']:
        values = pd.to_numeric(subset[col], errors='coerce').to_numpy()
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f'{col} must contain finite nonnegative values.')
    if (subset['cpu_demand_mi'] <= 0).any():
        raise ValueError('CPU demands must be positive.')
    for key in ['edge_nodes', 'cloud_servers']:
        if int(params[key]) < 1 or int(params[key]) != float(params[key]):
            raise ValueError(f'{key} must be a positive integer.')
    for key in ['edge_queue_capacity', 'cloud_queue_capacity']:
        if float(params[key]) < 0 or int(params[key]) != float(params[key]):
            raise ValueError(f'{key} must be a nonnegative integer.')
    for key in ['edge_service_rate_tps_each', 'cloud_service_rate_tps_each']:
        if not np.isfinite(float(params[key])) or float(params[key]) <= 0:
            raise ValueError(f'{key} must be finite and positive.')
    if params.get('network_model', 'shared_fcfs') not in ['independent', 'shared_fcfs']:
        raise ValueError('network_model must be independent or shared_fcfs.')
    for key in ['utilization_threshold', 'network_utilization_threshold']:
        value = float(params.get(key, 0.8))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f'{key} must be in [0,1].')
    for key, default in [('bandwidth_mbps', 25.0), ('network_probe_kb', 1024.0)]:
        if not np.isfinite(float(params.get(key, default))) or float(params.get(key, default)) <= 0:
            raise ValueError(f'{key} must be finite and positive.')
    if params.get('network_baseline_rule', 'paper_utilization') not in [
            'paper_utilization', 'reference_bandwidth']:
        raise ValueError('Unknown network baseline rule.')


DEADLINE_COMPONENTS = ('capacity_fallback', 'feasibility_override',
                       'proxy_tie_break', 'least_lateness')


def deadline_component_settings(components=None):
    """Explicit diagnostic switches; never part of physical environment state."""
    supplied = {} if components is None else dict(components)
    unknown = set(supplied).difference(DEADLINE_COMPONENTS)
    if unknown or any(type(value) is not bool for value in supplied.values()):
        raise ValueError('Components must be known names with boolean values.')
    return {name: supplied.get(name, True) for name in DEADLINE_COMPONENTS}


def apply_deadline_stages(selected, reason, *, eavailable, cavailable,
                          eresponse, cresponse, deadline, eservice, cservice,
                          payload_kb, params, ewait, threshold_ms, components):
    """Original ordered actions, with independently disabled diagnostic stages.

    Recording an applied branch does not imply it changed the destination.
    All True reproduces the submitted deadline-policy decision code verbatim in
    behavior. Removing a stage leaves other stages' eligibility unchanged.
    """
    diagnostic = {'baseline_destination': selected}
    for name in DEADLINE_COMPONENTS:
        diagnostic[name + '_applied'] = False
        diagnostic[name + '_changed_destination'] = False
    diagnostic['least_lateness_bypassed_edge_wait_gate'] = False
    old = selected
    if components['capacity_fallback'] and selected == 'cloud' and not cavailable:
        selected = 'edge' if eavailable else 'blocked'
        reason = 'predicted_capacity_fallback'
        diagnostic['capacity_fallback_applied'] = True
        diagnostic['capacity_fallback_changed_destination'] = selected != old
    diagnostic['after_capacity_destination'] = selected
    efeasible = eavailable and eresponse <= deadline + 1e-9
    cfeasible = cavailable and cresponse <= deadline + 1e-9
    diagnostic['predicted_edge_feasible'] = bool(efeasible)
    diagnostic['predicted_cloud_feasible'] = bool(cfeasible)
    old = selected
    if selected == 'edge' and not efeasible and cfeasible:
        if components['feasibility_override']:
            selected, reason = 'cloud', 'deadline_override_to_cloud'
            diagnostic['feasibility_override_applied'] = True
    elif selected == 'cloud' and not cfeasible and efeasible:
        if components['feasibility_override']:
            selected, reason = 'edge', 'deadline_override_to_edge'
            diagnostic['feasibility_override_applied'] = True
    elif efeasible and cfeasible:
        gap = abs(eresponse-cresponse)
        slack = deadline-min(eresponse, cresponse)
        if (components['proxy_tie_break']
                and gap <= float(params.get('deadline_tie_gap_ms', 25.0))
                and slack >= float(params.get('deadline_tie_slack_ms', 100.0))):
            ec, ee = cost_energy_proxy('edge', eservice, payload_kb, params)
            cc, ce = cost_energy_proxy('cloud', cservice, payload_kb, params)
            cw, ew = float(params.get('proxy_cost_weight', 1.0)), float(params.get('proxy_energy_weight', 1.0))
            selected = 'edge' if cw*ec+ew*ee <= cw*cc+ew*ce else 'cloud'
            reason = 'proxy_tie_break_to_' + selected
            diagnostic['proxy_tie_break_applied'] = True
    elif not efeasible and not cfeasible and selected != 'blocked':
        if components['least_lateness']:
            elate = eresponse-deadline if eavailable else np.inf
            clate = cresponse-deadline if cavailable else np.inf
            if elate < clate:
                selected, reason = 'edge', 'least_predicted_lateness'
                diagnostic['least_lateness_applied'] = True
            elif clate < elate:
                selected, reason = 'cloud', 'least_predicted_lateness'
                diagnostic['least_lateness_applied'] = True
    for name in ('feasibility_override', 'proxy_tie_break', 'least_lateness'):
        diagnostic[name + '_changed_destination'] = (
            diagnostic[name + '_applied'] and selected != old)
    diagnostic['least_lateness_bypassed_edge_wait_gate'] = bool(
        diagnostic['least_lateness_changed_destination']
        and selected == 'edge' and ewait > threshold_ms)
    diagnostic['after_deadline_stages_destination'] = selected
    return selected, reason, diagnostic


def simulate_subset(subset: pd.DataFrame, params: Dict, mode: str,
                    threshold_ms: float = HYBRID_EDGE_DELAY_THRESHOLD_MS,
                    diagnostic_components: Optional[Dict] = None):
    """Run one policy on a frozen workload with a common event/admission engine.

    At a timestamp: implicit service completions occur first, then previously
    dispatched cloud arrivals, then source arrivals in stable input order.
    A newly dispatched zero-delay cloud arrival precedes the next source event.
    The uplink is either the submitted independent-transfer assumption or a
    shared FCFS byte server. Propagation and the return-delay proxy do not hold
    the uplink. Neither option models packet loss or handovers.
    """
    _validate(subset, params, mode)
    components = deadline_component_settings(diagnostic_components)
    if diagnostic_components and mode != 'deadline_aware_hybrid':
        raise ValueError('Diagnostic component switches require the deadline policy.')
    subset = subset.sort_values('arrival_time_s', kind='stable').reset_index(drop=True)
    edge = make_empty_layer_state(int(params['edge_nodes']))
    cloud = make_empty_layer_state(int(params['cloud_servers']))
    eq, cq = int(params['edge_queue_capacity']), int(params['cloud_queue_capacity'])
    er, cr = float(params['edge_service_rate_tps_each']), float(params['cloud_service_rate_tps_each'])
    uplink = Uplink(float(params.get('bandwidth_mbps', 25.0)),
                    str(params.get('network_model', 'shared_fcfs')))
    tasks = subset.to_dict('records')
    events = [(float(t['arrival_time_s']) * 1000.0, 1, i, i)
              for i, t in enumerate(tasks)]
    heapq.heapify(events)
    inflight, rows = {}, {}
    end_ms = 0.0
    proposed = mode in ['hybrid_cloud_edge', 'deadline_aware_hybrid']

    def finish_row(i, destination, sched, diagnostic, arrival_ms,
                   upload_wait=0.0, upload_service=0.0):
        task = tasks[i]
        accepted = bool(sched['accepted'])
        service = service_time_ms(float(task['cpu_demand_mi']), er if destination == 'edge' else cr)
        ret = return_delay_ms(task['edge_cloud_network_delay_ms']) if destination == 'cloud' else 0.0
        finish = sched['service_finish_time_ms'] + ret if accepted else None
        response = finish - float(task['arrival_time_s']) * 1000.0 if accepted else None
        deadline = float(task['deadline_ms'])
        late = int(accepted and response > deadline + 1e-9)
        cost, energy = cost_energy_proxy(destination, service if accepted else 0.0,
                                         float(task['task_size_kb']), params)
        if not accepted and destination != 'cloud':
            cost, energy = 0.0, 0.0
        row = dict(task_id=task['task_id'], scenario=task['scenario'],
                   run_id=int(task['run_id']), seed=task.get('seed', np.nan), mode=mode,
                   arrival_time_s=float(task['arrival_time_s']),
                   processing_location=destination if accepted else 'blocked',
                   requested_destination=destination, status='completed' if accepted else 'blocked',
                   failure_reason='' if accepted else destination + '_queue_capacity',
                   deadline_ms=deadline, start_time_ms=sched['start_time_ms'],
                   finish_time_ms=finish, service_time_ms=service if accepted else None,
                   waiting_time_ms=sched['waiting_time_ms'], response_time_ms=response,
                   deadline_slack_ms=deadline-response if accepted else None,
                   sla_violation=late, on_time_completion=int(accepted and not late),
                   edge_queue_estimate=diagnostic['edge_queue_at_decision'],
                   cloud_queue_estimate=diagnostic['cloud_queue_at_decision'],
                   actual_queue_at_layer_arrival=sched['queue_length_at_arrival'],
                   cloud_arrival_time_ms=arrival_ms if destination == 'cloud' else None,
                   network_queue_wait_ms=upload_wait,
                   upload_serialization_ms=upload_service,
                   cloud_transfer_delay_ms=upload_wait+upload_service,
                   cloud_return_delay_ms=ret, cost_proxy=cost, energy_proxy=energy)
        row.update(diagnostic)
        estimate = diagnostic.get('predicted_' + destination + '_response_ms', np.nan)
        row['prediction_error_ms'] = response-estimate if accepted and np.isfinite(estimate) else np.nan
        rows[i] = row

    while events:
        now, kind, order, i = heapq.heappop(events)
        end_ms = max(end_ms, now)
        purge_started_waiting_tasks(edge, now)
        purge_started_waiting_tasks(cloud, now)
        task = tasks[i]
        if kind == 0:
            item = inflight.pop(i)
            sched = schedule_on_layer(cloud, now, item['service_ms'], cq)
            finish_row(i, 'cloud', sched, item['diagnostic'], now,
                       item['upload_wait'], item['upload_service'])
            continue

        equeue, cqueue = waiting_queue_length(edge, now), waiting_queue_length(cloud, now)
        ebusy, cbusy = busy_servers(edge, now), busy_servers(cloud, now)
        utilization = ebusy / len(edge.server_available_times)
        probe_kb = float(params.get('network_probe_kb', 1024.0))
        if params.get('network_baseline_rule') == 'reference_bandwidth':
            probe_kb = 128.0
        available_bw, wan_util = uplink.probe(now, probe_kb)
        eservice = service_time_ms(float(task['cpu_demand_mi']), er)
        cservice = service_time_ms(float(task['cpu_demand_mi']), cr)
        ewait = estimate_waiting_delay_ms(edge, now)
        eresponse = ewait + eservice
        upload_wait, upload_service = uplink.predict(now, float(task['task_size_kb']))
        nominal_network = float(task['edge_cloud_network_delay_ms'])
        carrival = now + upload_wait + upload_service + nominal_network
        eavailable = can_accept(edge, now, eq)
        cwait, forecast_queue, cavailable, cresponse = np.nan, np.nan, True, np.nan
        if proposed:
            cwait, forecast_queue, cavailable = forecast_cloud(cloud, inflight, carrival, cq)
            cresponse = (carrival-now + cwait + cservice + return_delay_ms(nominal_network))
        d = dict(edge_queue_at_decision=equeue, cloud_queue_at_decision=cqueue,
                 cpu_demand_mi=float(task['cpu_demand_mi']),
                 task_size_kb=float(task['task_size_kb']),
                 edge_cloud_network_delay_ms=nominal_network,
                 predicted_edge_available=eavailable,
                 edge_busy_servers_at_decision=ebusy, cloud_busy_servers_at_decision=cbusy,
                 known_inflight_cloud_tasks=len(inflight),
                 edge_utilization_at_decision=utilization,
                 estimated_wan_utilization=wan_util,
                 wan_available_bandwidth_estimate_mbps=available_bw,
                 predicted_edge_wait_ms=ewait if proposed else np.nan,
                 predicted_cloud_wait_ms=cwait,
                 predicted_cloud_queue_at_arrival=forecast_queue,
                 predicted_edge_response_ms=eresponse if proposed else np.nan,
                 predicted_cloud_response_ms=cresponse,
                 predicted_cloud_available=cavailable if proposed else None,
                 network_model=uplink.model)

        if mode == 'edge_only':
            selected, reason = 'edge', 'fixed_edge'
        elif mode == 'cloud_only':
            selected, reason = 'cloud', 'fixed_cloud'
        elif mode == 'utilization_based':
            selected = 'cloud' if utilization > float(params.get('utilization_threshold', 0.8)) else 'edge'
            reason = 'edge_busy_fraction_threshold'
        elif mode == 'network_aware':
            if params.get('network_baseline_rule', 'paper_utilization') == 'paper_utilization':
                selected = 'cloud' if wan_util < float(params.get('network_utilization_threshold', 0.8)) else 'edge'
                reason = 'paper_network_utilization_rule_adapted'
            else:
                # The public reference calculates 1/probe_delay for its nominal
                # one-Mbit probe, represented in that code by 128 KB.
                probe_wait, probe_service = uplink.predict(now, 128.0)
                reference_bw = 1000.0 / (probe_wait + probe_service)
                selected = 'cloud' if reference_bw > float(params.get('network_bandwidth_threshold_mbps', 6.0)) else 'edge'
                reason = 'public_reference_bandwidth_rule_adapted'
                d['reference_probe_bandwidth_mbps'] = reference_bw
        elif mode == 'queue_length_based':
            edge_load = (equeue+ebusy) / len(edge.server_available_times)
            cloud_load = (cqueue+cbusy) / len(cloud.server_available_times)
            selected = 'edge' if edge_load <= cloud_load else 'cloud'
            reason = 'minimum_current_jobs_per_server'
        else:
            use_edge = eavailable and ewait <= threshold_ms and eresponse <= cresponse
            selected = 'edge' if use_edge else 'cloud'
            reason = 'hybrid_queue_and_response_rule'
            d['baseline_destination'] = selected
            if mode == 'deadline_aware_hybrid':
                selected, reason, stages = apply_deadline_stages(
                    selected, reason, eavailable=eavailable, cavailable=cavailable,
                    eresponse=eresponse, cresponse=cresponse,
                    deadline=float(task['deadline_ms']), eservice=eservice,
                    cservice=cservice, payload_kb=float(task['task_size_kb']),
                    params=params, ewait=ewait, threshold_ms=threshold_ms,
                    components=components)
                d.update(stages)
        d['decision_reason'] = reason

        if selected == 'edge':
            sched = schedule_on_layer(edge, now, eservice, eq)
            finish_row(i, 'edge', sched, d, now)
        elif selected == 'cloud':
            upload_wait, upload_service, upload_end = uplink.submit(now, float(task['task_size_kb']))
            arrival = upload_end + nominal_network
            inflight[i] = dict(arrival_ms=arrival, order=order, service_ms=cservice,
                               upload_wait=upload_wait, upload_service=upload_service,
                               diagnostic=d)
            heapq.heappush(events, (arrival, 0, order, i))
        else:
            sched = dict(accepted=False, start_time_ms=None, service_finish_time_ms=None,
                         waiting_time_ms=None, queue_length_at_arrival=equeue)
            finish_row(i, 'blocked', sched, d, now)
            rows[i]['failure_reason'] = 'both_tiers_predicted_full'

    if len(rows) != len(tasks) or inflight:
        raise AssertionError('Task conservation failed.')
    detailed = pd.DataFrame([rows[i] for i in range(len(tasks))])
    summary = summarize_results(detailed, edge, cloud, subset, mode, params, end_ms, uplink)
    return detailed, summary


def summarize_results(detailed_rows, edge_state: LayerState, cloud_state: LayerState,
                      subset: pd.DataFrame, mode: str, params: Optional[Dict] = None,
                      last_event_ms: float = 0.0, uplink: Optional[Uplink] = None) -> Dict:
    frame = pd.DataFrame(detailed_rows)
    completed = frame[frame['status'] == 'completed']
    n, nc = len(frame), len(completed)
    nb = int((frame['status'] == 'blocked').sum())
    late = int(completed['sla_violation'].sum())
    ontime = nc-late
    finish = float(completed['finish_time_ms'].max()) if nc else 0.0
    horizon_ms = max(float(subset['arrival_time_s'].max())*1000.0, 1.0)
    input_horizon = float((params or {}).get('duration_s', horizon_ms/1000.0))
    end_ms = max(finish, horizon_ms, last_event_ms, input_horizon*1000.0, 1.0)
    environment_hash = hashlib.sha256(json.dumps(params or {}, sort_keys=True,
                                                  default=str).encode()).hexdigest()
    cost, energy = float(frame['cost_proxy'].sum()), float(frame['energy_proxy'].sum())
    return dict(
        scenario=str(subset['scenario'].iloc[0]), run_id=int(subset['run_id'].iloc[0]),
        seed=subset['seed'].iloc[0] if 'seed' in subset else np.nan, mode=mode,
        network_model=(params or {}).get('network_model', 'shared_fcfs'),
        environment_sha256=environment_hash,
        workload_sha256=workload_fingerprint(subset), total_tasks=n,
        completed_tasks=nc, blocked_tasks=nb, sla_violations=late,
        on_time_completed_tasks=ontime, unsuccessful_tasks=n-ontime,
        on_time_completion_rate=ontime/n, unsuccessful_task_rate=(n-ontime)/n,
        throughput_tasks_per_s=nc/(end_ms/1000.0),
        offered_rate_tps=n/input_horizon if input_horizon > 0 else np.nan,
        observation_duration_s=input_horizon, drain_end_s=end_ms/1000.0,
        avg_response_time_ms=float(completed['response_time_ms'].mean()) if nc else np.nan,
        p95_response_time_ms=float(completed['response_time_ms'].quantile(0.95)) if nc else np.nan,
        avg_waiting_time_ms=float(completed['waiting_time_ms'].mean()) if nc else np.nan,
        avg_network_queue_wait_ms=float(frame['network_queue_wait_ms'].mean()),
        edge_utilization=edge_state.total_busy_time_ms/(len(edge_state.server_available_times)*end_ms),
        cloud_utilization=cloud_state.total_busy_time_ms/(len(cloud_state.server_available_times)*end_ms),
        avg_edge_queue_length=compute_average_queue_length(edge_state.queue_events, end_ms),
        avg_cloud_queue_length=compute_average_queue_length(cloud_state.queue_events, end_ms),
        blocking_probability=nb/n,
        cloud_offloading_ratio=float((completed['processing_location']=='cloud').sum())/nc if nc else 0.0,
        cloud_routing_fraction=float((frame['requested_destination']=='cloud').sum())/n,
        sla_violation_rate=late/nc if nc else np.nan,
        avg_cost_proxy=cost/nc if nc else np.nan, total_cost_proxy=cost,
        avg_energy_proxy=energy/nc if nc else np.nan, total_energy_proxy=energy,
        cost_proxy_per_offered_task=cost/n, energy_proxy_per_offered_task=energy/n,
        uplink_utilization=(uplink.total_busy_ms/end_ms
                            if uplink and uplink.model=='shared_fcfs' else np.nan),
        cloud_forecasts_with_positive_wait=int((frame['predicted_cloud_wait_ms'].fillna(0)>0).sum()),
    )


def simulate_edge_only(subset, params):
    return simulate_subset(subset, params, 'edge_only')


def simulate_cloud_only(subset, params):
    return simulate_subset(subset, params, 'cloud_only')


def simulate_hybrid(subset, params, threshold_ms=HYBRID_EDGE_DELAY_THRESHOLD_MS):
    return simulate_subset(subset, params, 'hybrid_cloud_edge', threshold_ms)


def simulate_deadline_aware_hybrid(subset, params):
    return simulate_subset(subset, params, 'deadline_aware_hybrid')
