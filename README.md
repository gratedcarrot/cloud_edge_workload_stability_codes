Cloud-Edge Workload Stability
This repository contains Python code for reproducing the experiments in the paper:
Performance Analysis of a Probabilistic Model for Large-Scale Workload Handling in Cloud-Edge Systems
The code generates controlled cloud-edge workload traces, runs edge-only, cloud-only, and hybrid cloud-edge simulations, performs sensitivity analysis, and runs a public trace-derived validation using Google ClusterData 2011-2.
Files
File	Purpose
`generate_workload.py`	Generates controlled synthetic workloads for low, medium, high, burst, and extended scenarios.
`run_simulation.py`	Runs controlled simulations and produces summary outputs with 95% confidence intervals.
`run_sensitivity.py`	Runs sensitivity analysis only, using the generated workload.
`build_public_trace_data.py`	Downloads and maps Google ClusterData 2011-2 task events into the simulator input format.
`normalize_public_trace.py`	Normalizes public trace timing to the validation window used in the study.
`run_public_trace_validation.py`	Runs edge-only, cloud-only, and hybrid simulation on the public trace-derived dataset.
Installation
```bash
pip install -r requirements.txt
```
Python 3.10 or newer is recommended.
Controlled experiments
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
Or run sensitivity analysis separately after generating the workload:
```bash
python run_sensitivity.py
```
Main outputs are written to:
```text
generated_workload/
simulation_results/
```
Public trace-derived validation
Build the public trace-derived dataset:
```bash
python build_public_trace_data.py
```
Normalize trace timing:
```bash
python normalize_public_trace.py
```
Run validation:
```bash
python run_public_trace_validation.py
```
Outputs are written to:
```text
public_trace_data/
public_trace_results/
```
Notes
The Google ClusterData 2011-2 trace is not redistributed in this repository. The script `build_public_trace_data.py` downloads the required public trace files from the Google Cluster Data Repository and maps task submission events into the simulator format.
The generated CSV outputs and figures can be regenerated from the scripts above.
