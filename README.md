# Cloud-Edge Workload Stability Codes

This repository contains the Python source code used for the paper:

**Performance Analysis of a Probabilistic Model for Large-Scale Workload Handling in Cloud-Edge Systems**

The code evaluates four processing modes:

1. `edge_only`
2. `cloud_only`
3. `hybrid_cloud_edge`
4. `deadline_aware_hybrid`

It supports controlled workload generation, controlled simulation, sensitivity analysis, public trace preprocessing, and public trace-derived validation. Cost and energy values are reported as normalized proxy metrics, not real billing or measured energy values.

## Repository contents

| File | Purpose |
|---|---|
| `generate_workload.py` | Generates controlled synthetic workloads for low, medium, high, burst, and extended scenarios. |
| `run_simulation.py` | Runs controlled simulations for all four modes and produces summaries, confidence intervals, statistical tests, and plots. |
| `run_sensitivity.py` | Runs sensitivity analysis separately after controlled workloads are generated. |
| `build_public_trace_data.py` | Builds the public trace-derived validation workload from Google ClusterData 2011-2 task events. |
| `normalize_public_trace.py` | Normalizes public trace timing to the validation window and refreshes the public trace summary. |
| `run_public_trace_validation.py` | Runs validation on the public trace-derived dataset using the same simulator logic. |
| `requirements.txt` | Python dependencies. |
| `RUN_ORDER.txt` | Recommended execution order. |
| `.gitignore` | Excludes generated data, results, figures, caches, and raw trace downloads. |

## Installation

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
```

## Recommended run order

```bash
python generate_workload.py
python run_simulation.py --no-detailed
python run_simulation.py --sensitivity --no-detailed
python build_public_trace_data.py
python normalize_public_trace.py
python run_public_trace_validation.py
```

The `--no-detailed` flag avoids creating the very large task-level controlled result file. Paper-level summaries and figures can still be generated without it.

## Optional sensitivity command

By default, sensitivity analysis is run for the baseline hybrid mode. To also include deadline-aware hybrid in capacity/network sensitivity sweeps, use:

```bash
python run_simulation.py --sensitivity --sensitivity-all-modes --no-detailed
```

## Generated folders

The following folders are generated when the scripts are run and are intentionally excluded from the repository:

```text
generated_workload/
simulation_results/
public_trace_data/
public_trace_results/
trace_downloads/
```

## Public trace note

The Google ClusterData 2011-2 trace is not redistributed in this repository. The script `build_public_trace_data.py` downloads the required public trace files from the Google Cluster Data Repository and maps task submission events into the cloud-edge simulation format. The public trace is used for workload-pattern validation, not as a native cloud-edge trace.

## License

This code is released under the MIT License.
