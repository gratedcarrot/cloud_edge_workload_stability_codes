from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from run_simulation import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    TASK_FILE_NAME,
    PARAM_FILE_NAME,
    service_time_ms,
)


SCENARIOS = ["low_load", "medium_load", "high_load"]

SCENARIO_LABELS = {
    "low_load": "Low",
    "medium_load": "Medium",
    "high_load": "High",
}

SIM_SUMMARY_FILE = "controlled_summary_with_ci.csv"


def get_scenario_row(params_df: pd.DataFrame, scenario: str) -> pd.Series:
    row = params_df[params_df["scenario"] == scenario]
    if row.empty:
        raise ValueError(f"No scenario parameters found for: {scenario}")
    return row.iloc[0]


def get_arrival_rate(params_df: pd.DataFrame, scenario: str) -> float:
    row = get_scenario_row(params_df, scenario)

    for col in ["base_lambda_tps", "arrival_rate_tps", "arrival_rate", "lambda_tps"]:
        if col in row.index and pd.notna(row[col]):
            return float(row[col])

    raise KeyError(f"Arrival-rate column not found for scenario: {scenario}")


def mmck_metrics(lam: float, mu: float, c: int, k: int) -> Dict[str, float]:
    if lam <= 0:
        raise ValueError("Arrival rate must be positive.")
    if mu <= 0:
        raise ValueError("Service rate must be positive.")
    if c <= 0:
        raise ValueError("Number of servers must be positive.")
    if k < c:
        raise ValueError("Total capacity K must be greater than or equal to c.")

    a = lam / mu
    terms: List[float] = []

    for n in range(k + 1):
        if n < c:
            term = (a ** n) / math.factorial(n)
        else:
            term = (a ** n) / (math.factorial(c) * (c ** (n - c)))
        terms.append(term)

    p0 = 1.0 / sum(terms)
    probabilities = [term * p0 for term in terms]

    p_block = probabilities[k]
    lambda_eff = lam * (1.0 - p_block)

    l_system = sum(n * probabilities[n] for n in range(k + 1))
    l_queue = sum(max(0, n - c) * probabilities[n] for n in range(k + 1))

    response_time_ms = (l_system / lambda_eff) * 1000.0 if lambda_eff > 0 else np.nan
    waiting_time_ms = (l_queue / lambda_eff) * 1000.0 if lambda_eff > 0 else np.nan

    return {
        "analytical_block_probability": p_block,
        "analytical_throughput": lambda_eff,
        "analytical_response_time_ms": response_time_ms,
        "analytical_waiting_time_ms": waiting_time_ms,
    }


def fmt_ci(mean: float, ci: float, multiplier: float = 1.0, decimals: int = 2) -> str:
    if not np.isfinite(mean):
        return "NA"
    return f"{mean * multiplier:.{decimals}f} ± {ci * multiplier:.{decimals}f}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = []

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():
        values = [str(row[col]) for col in headers]
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def build_validation_table(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    task_file = input_dir / TASK_FILE_NAME
    param_file = input_dir / PARAM_FILE_NAME
    sim_summary_file = output_dir / SIM_SUMMARY_FILE

    if not task_file.exists():
        raise FileNotFoundError(f"Missing workload trace: {task_file}")
    if not param_file.exists():
        raise FileNotFoundError(f"Missing scenario parameter file: {param_file}")
    if not sim_summary_file.exists():
        raise FileNotFoundError(
            f"Missing simulation summary: {sim_summary_file}. "
            "Run run_simulation.py before analytical_validation.py."
        )

    tasks_df = pd.read_csv(task_file)
    params_df = pd.read_csv(param_file)
    sim_df = pd.read_csv(sim_summary_file)

    rows = []

    for scenario in SCENARIOS:
        scenario_tasks = tasks_df[tasks_df["scenario"] == scenario].copy()
        if scenario_tasks.empty:
            continue

        params = get_scenario_row(params_df, scenario)

        edge_servers = int(params["edge_nodes"])
        edge_queue_capacity = int(params["edge_queue_capacity"])
        total_capacity = edge_servers + edge_queue_capacity

        edge_service_rate = float(params["edge_service_rate_tps_each"])
        arrival_rate = get_arrival_rate(params_df, scenario)

        service_times = scenario_tasks["cpu_demand_mi"].apply(
            lambda cpu: service_time_ms(cpu, edge_service_rate)
        )

        effective_mu = 1000.0 / float(service_times.mean())

        analytical = mmck_metrics(
            lam=arrival_rate,
            mu=effective_mu,
            c=edge_servers,
            k=total_capacity,
        )

        sim_row = sim_df[
            (sim_df["scenario"] == scenario) & (sim_df["mode"] == "edge_only")
        ]

        if sim_row.empty:
            raise ValueError(f"Missing edge_only simulation result for: {scenario}")

        sim_row = sim_row.iloc[0]

        rows.append(
            {
                "Scenario": SCENARIO_LABELS.get(scenario, scenario),
                "Arrival rate λ": round(arrival_rate, 2),
                "Effective service rate μ": round(effective_mu, 4),
                "Analytical block (%)": round(
                    analytical["analytical_block_probability"] * 100.0, 2
                ),
                "Simulated block (%)": fmt_ci(
                    sim_row["blocking_probability_mean"],
                    sim_row["blocking_probability_ci95"],
                    multiplier=100.0,
                    decimals=2,
                ),
                "Analytical throughput": round(
                    analytical["analytical_throughput"], 2
                ),
                "Simulated throughput": fmt_ci(
                    sim_row["throughput_tasks_per_s_mean"],
                    sim_row["throughput_tasks_per_s_ci95"],
                    decimals=2,
                ),
                "Analytical response time (ms)": round(
                    analytical["analytical_response_time_ms"], 2
                ),
                "Simulated response time (ms)": fmt_ci(
                    sim_row["avg_response_time_ms_mean"],
                    sim_row["avg_response_time_ms_ci95"],
                    decimals=2,
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare analytical M/M/c/K estimates with edge-only simulation results."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Folder containing generated workload files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder containing simulation output files.",
    )
    args = parser.parse_args()

    table = build_validation_table(args.input_dir, args.output_dir)

    output_csv = args.output_dir / "analytical_vs_simulation_validation.csv"
    output_md = args.output_dir / "analytical_vs_simulation_validation.md"

    table.to_csv(output_csv, index=False)
    output_md.write_text(dataframe_to_markdown(table), encoding="utf-8")

    print("\nAnalytical vs simulation validation table:\n")
    print(table.to_string(index=False))
    print("\nCreated:")
    print(output_csv)
    print(output_md)


if __name__ == "__main__":
    main()
