# Cloud-Edge Workload Stability

This repository contains Python code for reproducing the experiments in the paper:

**Performance Analysis of a Probabilistic Model for Large-Scale Workload Handling in Cloud-Edge Systems**

The code generates controlled cloud-edge workload traces, runs edge-only, cloud-only, and hybrid cloud-edge simulations, performs sensitivity analysis, and runs a public trace-derived validation using Google ClusterData 2011-2.

## Files

| File                             | Purpose                                                                                                                   |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `generate_workload.py`           | Generates controlled synthetic workloads for low, medium, high, burst, and extended workload scenarios.                   |
| `run_simulation.py`              | Runs controlled edge-only, cloud-only, and hybrid simulations and produces summary outputs with 95% confidence intervals. |
| `run_sensitivity.py`             | Runs sensitivity analysis using the generated workload.                                                                   |
| `build_public_trace_data.py`     | Downloads and maps Google ClusterData 2011-2 task events into the simulator input format.                                 |
| `normalize_public_trace.py`      | Normalizes public trace timing to the validation window used in the study.                                                |
| `run_public_trace_validation.py` | Runs edge-only, cloud-only, and hybrid simulation on the public trace-derived dataset.                                    |
| `requirements.txt`               | Lists the Python dependencies required to run the scripts.                                                                |

## Installation

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
```

## Controlled Experiments

Generate controlled workload traces:

```bash
python generate_workload.py
```

Run the main controlled simulations:

```bash
python run_simulation.py
```

Run controlled simulations with sensitivity analysis:

```bash
python run_simulation.py --sensitivity --no-detailed
```

Alternatively, run only sensitivity analysis after generating the workload:

```bash
python run_sensitivity.py
```

## Public Trace-Derived Validation

Build the public trace-derived dataset:

```bash
python build_public_trace_data.py
```

Normalize the public trace timing:

```bash
python normalize_public_trace.py
```

Run public trace validation:

```bash
python run_public_trace_validation.py
```

## Outputs

The scripts generate output folders automatically, including:

* controlled workload traces
* simulation summaries
* confidence-interval tables
* statistical test outputs
* sensitivity analysis results
* public trace validation results

Large raw public trace files are not redistributed in this repository. The public validation uses Google ClusterData 2011-2, which is available from the Google Cluster Data Repository.

## Reproducibility Notes

The controlled workload generator uses fixed random seeds for reproducibility. Each workload scenario is generated across 30 independent runs by default. The same generated workload trace is evaluated under edge-only, cloud-only, and hybrid processing modes to ensure fair comparison.

## License

This repository is provided for academic and research use.
