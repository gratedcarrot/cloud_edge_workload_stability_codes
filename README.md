# Cloud–Edge Workload Stability

This repository contains the simulation code, input datasets, statistical analyses and numerical results for our manuscript, **“Performance Analysis of a Probabilistic Model for Large-Scale Workload Handling in Cloud-Edge Systems.”**

**Authors:** Aryan Paode and Minirani S., SVKM’s NMIMS, Mumbai, India.

We study task routing between finite-capacity edge and cloud queues. Our evaluation considers response time, blocking, deadline attainment, offloading, and cost and energy proxies. Alongside the queue-aware and deadline-aware policies, we evaluate simpler utilization-, network- and queue-length-based rules. We report conditions in which these simpler rules perform better; deadline awareness describes the information used in routing, not a guarantee of improved performance.

## Repository contents

```text
engine/         Common simulator, C++17 acceleration and Python reference model
scripts/        Experiment, analysis, table-export and plotting programs
data/           Fixed synthetic inputs, Google trace subset and EUA coordinates
results/        Reference run-level results and numerical tables
figures/        Manuscript Figures 1–5 in PNG, PDF and SVG formats
tests/          Queue, routing, deadline and trace-extraction regression tests
validation/     Recorded verification results and execution environment
run.py          Entry point for reproduction commands
requirements.txt
CHECKSUMS.sha256
LICENSE
CITATION.cff
README.md
```

`data/` and `results/` are part of the research record and are intentionally tracked. Rerunning an experiment writes to `outputs/`, leaving the supplied results unchanged.

## Installation

Use Python 3.12 or newer and install the pinned packages:

```bash
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS, or `.\.venv\Scripts\Activate.ps1` in Windows PowerShell, then run:

```bash
python -m pip install -r requirements.txt
```

The full workflow also needs a C++17 compiler: `g++` or `clang++` on `PATH`, or an executable specified through `CXX`. The compiled libraries are cached in the system temporary directory, not committed to the repository. Linux was tested with Python 3.13.5 and GNU C++ 14.2. Native Windows execution has not been tested; Windows users can run the full workflow in WSL with a compatible compiler.

Inspecting the saved results, exporting tables, and regenerating figures do not require a C++ compiler. The simulation API also includes a slower Python reference backend. The complete experiment drivers use the compiled backend where specified.

## Getting started

Run these commands from the repository folder:

```bash
# Check file hashes, input identity, result dimensions and task accounting.
python run.py verify

# Run regression tests and compare C++ outcomes with the Python reference.
python run.py test

# Replay one complete synthetic workload and selected Google/locality cases.
python run.py smoke

# Recreate numerical tables and manuscript figures from saved results.
python run.py tables
python run.py figures
```

Generated tables are written to `outputs/tables/`; regenerated figures are written to `outputs/figures/`. Figures retain the manuscript’s metric definitions, axes and confidence intervals. Short intervals can fall inside their markers.

## Reproducing the experiments

The complete workflow is:

```bash
python run.py all --workers 4
```

It checks the inputs, runs the tests and experiments, performs the analyses, exports the tables, regenerates the figures, and compares the regenerated result CSVs with the supplied reference results. Decrease `--workers` for a machine with fewer available cores. A separate output directory can be selected with `--output PATH`.

Individual stages can also be run:

| Command | Experiment or output |
|---|---|
| `python run.py main --workers 4` | 3,600 main evaluations: five scenarios × 30 workloads × three network settings × eight policies; primary paired analysis |
| `python run.py sensitivity --workers 4` | 1,200 threshold/propagation and 1,440 capacity/buffer evaluations on matched high/burst inputs |
| `python run.py trace --workers 4` | 720 mapping evaluations of the same 20,000 Google submissions |
| `python run.py proxy --workers 4` | 1,248 Google coefficient evaluations; a separate six-workload synthetic coefficient illustration |
| `python run.py locality --workers 4` | 3,360 policy/condition evaluations on 30 matched synthetic workloads with static EUA-based eligibility |
| `python run.py analytical` | 90 isolated M/M/4/12 replications and distribution diagnostics on the study inputs |
| `python run.py tables` | Main and supplementary numerical table exports, four-policy tests and secondary baseline contrasts |
| `python run.py figures` | Figures 1–5 from the numerical summaries |
| `python run.py compare` | Numerical comparison of a complete rerun against the reference result CSVs |

Repeated policies, mappings or link conditions do not create independent workloads. The main and locality studies use 30 independent workloads per reported scenario. The Google study uses one fixed window, so its sensitivity results are descriptive rather than independent replications.

To regenerate the 150 synthetic input files from the fixed random seeds, run `python run.py generate`. This writes to `outputs/data/` rather than replacing `data/synthetic/`. Generation preserves the original random-draw order, including descriptive marks that are not used by the routing rules.

All ordinary reproduction commands run offline using the supplied inputs. Re-downloading the Google shard is optional:

```bash
python scripts/build_public_trace_data.py --target-records 20000 --output-dir outputs/google_download --max-source-mb 64 --max-uncompressed-mb 512 --max-rows 3000000 --max-seconds 120
```

## Policies and common assumptions

| Code identifier | Routing rule |
|---|---|
| `edge_only` | Use only the finite-capacity edge tier. |
| `cloud_only` | Upload to the cloud and test admission at physical cloud arrival. |
| `hybrid_cloud_edge` | Use edge when admissible, predicted edge wait is at most 220 ms, and predicted edge response is no greater than cloud response. |
| `deadline_aware_hybrid` | Add capacity fallback, deadline feasibility, a feasible proxy tie-break and the less-late choice when both routes are predicted late. |
| `deadline_guarded` | Omit only that final less-late choice; retain the other deadline-policy stages. |
| `utilization_based` | Select cloud when the fraction of busy edge servers exceeds 80%. |
| `network_aware` | Select cloud when the congestion estimate from a virtual 1,024-kB upload is below 80%. |
| `queue_length_based` | Choose the tier with fewer current jobs per server; break ties toward edge. |

The utilization and network rules adapt Algorithm 1 of Belcastro et al., *Edge-cloud continuum solutions for urban mobility prediction and planning*, IEEE Access 11 (2023), 38864–38874, DOI [10.1109/ACCESS.2023.3267471](https://doi.org/10.1109/ACCESS.2023.3267471). Our implementation uses a pooled edge tier and an upload-only virtual probe; it does not reproduce that paper’s mobility application or bidirectional network measurements.

Within a matched condition, every policy receives identical tasks, service rates, server counts, buffers and network assumptions. The standard configuration has four edge servers, three cloud servers, and 120/250 waiting places. Fixed-tier policies can use only their designated tier.

The three link settings are independent uploads at 25 Mbit/s, a shared first-come-first-served uplink at 25 Mbit/s, and a shared uplink at 250 Mbit/s. Independent uploads do not impose one aggregate bandwidth budget. Shared uploads serialize before propagation; return traffic does not compete for this uplink. The model does not simulate packet loss, retries, outages, handovers or data-residency restrictions.

## Units, metrics and statistical analyses

Arrival timestamps are in **seconds**. Service times, response times, waits, propagation and relative deadline budgets are in **milliseconds**. Payload is in **decimal kB** and link capacity in **Mbit/s**, giving serialization time `8 × payload_kB / bandwidth_Mbps` milliseconds. Proxy upload charges use decimal MB (`kB / 1000`). The explicit 1024-divisor sensitivity is retained for comparison; it is not the default convention.

The primary endpoint is **on-time completions / all offered tasks**. Blocking uses all offers, while completed-task SLA violations use **late completions / completions**. Response-time statistics cover completed tasks. Throughput uses the larger of the arrival horizon and the last drained event. Uploaded data are charged even when a task is rejected after reaching the cloud. Cost and energy are assumed scores, not currency, measured power or joules.

The primary analysis has 75 paired guarded-minus-comparator contrasts, pointwise 95% t intervals, sign-flip tests and Holm adjustment. Secondary comparisons of the two focal hybrid policies against the three adaptive baselines form a separate family of 90 contrasts.

**Table VIII compares four focal policies, not all eight.** It contains eight metrics × five scenarios: 40 repeated-measures ANOVA comparisons with Greenhouse–Geisser correction and 40 Friedman comparisons, with separate Holm adjustments. An identically zero blocking cell is entered with p = 1. The corresponding data are in `results/tables/secondary_four_policy_tests.csv`.

## Where to find the paper’s results

| Manuscript material | Reference output |
|---|---|
| Table II and analytical checks | `results/analytical/` |
| Table VII and the main policy comparison | `results/main/`, `results/tables/main_original_metrics_ci.csv` |
| Table VIII | `results/tables/secondary_four_policy_tests.csv` |
| Tables IX–X and Figures 4–5 | `results/sensitivity/`, `results/tables/sensitivity_original_metrics_ci.csv` |
| Table XII and Supplementary Table S1 | `results/google/base_outcomes_20k.csv` |
| Supplementary Table S2 | `results/google/mapping_sensitivity_20k.csv` |
| Supplementary Table S3 | `results/google/coefficient_sensitivity_20k.csv` |
| Supplementary Table S4 and the locality discussion | `results/locality/` |

Additional `table_*.csv` files in `results/tables/` provide convenient numerical exports. The source result files retain individual realizations and all tested settings. Derived table exports use percentage points where named; raw simulation rates use fractions.

## Data sources and limitations

**Synthetic workloads.** We supply 150 fixed inputs and their seeds and hashes. The scenarios are low, medium, high, burst and extended load. A separate six-input coefficient study is kept in `data/synthetic_proxy/`; it is not the 30-run main-study dataset.

**Google ClusterData 2011-2.** We retain the first 20,000 eligible submit events from shard 00000 at or after timestamp 600,000,000 microseconds. The native relative span is 1,535.418213 seconds. Rescaling that span to 600 seconds is a load assumption, not a unit conversion. Resource requests are mapped to modeled compute demand, payload and deadlines; these are not observed edge-task measurements. The 2,000-task files in `data/diagnostic_google/` are a nested prefix used only for the analytical-fit diagnostic, not an independent trace. The field mappings, exclusions and checksums are recorded in `data/google/metadata.json` and `extraction_provenance.json`.

Source: [Google ClusterData2011_2 documentation](https://github.com/google/cluster-data/blob/master/ClusterData2011_2.md). The original trace and documentation are provided by Google under CC-BY; retain the source attribution when reusing this subset.

**EUA.** The Melbourne CBD subset contains 125 candidate sites based on radio-base-station coordinates and 816 generated user positions. We select four active sites deterministically, keep total edge compute and buffer capacity fixed, and compare pooled and separate local queues under assumed radii. Geographic eligibility is not a measurement of radio coverage, mobility or data residency. Source links, the pinned commit and preprocessing details are in `data/eua/eua_preparation_report.json`.

Source: [Phu Lai et al., EUA Datasets](https://github.com/PhuLai/eua-dataset/tree/61238e00a6f01019c8a6ba1619675337a879a3d4). The dataset credits the Australian Communications and Media Authority for station locations. These third-party inputs retain their source ownership and terms; they are not relicensed under our software license.

## License and citation

Our source code is released under the MIT License in `LICENSE`. Public input datasets retain their original attribution and terms. Please cite the accompanying manuscript and identify the exact repository commit used in an experiment. `CITATION.cff` supplies the software authorship metadata.
