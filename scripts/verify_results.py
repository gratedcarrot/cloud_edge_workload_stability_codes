"""Check data integrity, matched comparisons, task accounting and rerun equivalence."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from _paths import ROOT, DATA, RESULTS, REFERENCE_RESULTS

COMPARISONS = {
    'main/runs.csv': ['scenario','run_id','network_model','bandwidth_mbps','mode'],
    'main/paired_primary.csv': ['scenario','network_model','bandwidth_mbps','candidate','comparator'],
    'main/policy_summary.csv': ['scenario','network_model','bandwidth_mbps','mode'],
    'sensitivity/sweep_runs.csv': ['scenario','run_id','network_model','parameter','level'],
    'sensitivity/capacity_runs.csv': ['scenario','run_id','network_model','parameter','level'],
    'google/base_outcomes_20k.csv': ['anchor','network_model','bandwidth_mbps','mode'],
    'google/mapping_sensitivity_20k.csv': ['anchor','mapping','network_model','bandwidth_mbps','mode'],
    'google/coefficient_sensitivity_20k.csv': ['anchor','coefficient_case','network_model','bandwidth_mbps','mode'],
    'synthetic_proxy/proxy_runs.csv': ['block','setting','network_model','bandwidth_mbps','mode'],
    'locality/run_results.csv': ['run','bandwidth_mbps','topology','radius_m','policy'],
    'locality/condition_means.csv': ['bandwidth_mbps','topology','radius_m','policy'],
    'locality/paired_contrasts.csv': ['radius_m','contrast','bandwidth_mbps','policy'],
    'analytical/mmck_runs.csv': ['rho','run'],
    'analytical/mmck_comparison.csv': ['rho','metric'],
    'analytical/mmck_state_probabilities.csv': ['rho','state'],
    'analytical/workload_assumptions_runs.csv': ['scenario','run'],
    'analytical/workload_assumptions_summary.csv': ['scenario','metric'],
    'tables/secondary_four_policy_tests.csv': ['scenario','metric'],
    'tables/secondary_baselines.csv': ['scenario','network_model','bandwidth_mbps','hybrid','baseline'],
    'tables/main_original_metrics_ci.csv': ['scenario','mode','network_model','bandwidth_mbps','metric'],
    'tables/sensitivity_original_metrics_ci.csv': ['scenario','network_model','parameter','level','metric'],
}


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def check_accounting(frame, name):
    require((frame.total_tasks == frame.completed_tasks + frame.blocked_tasks).all(), name+' total')
    require((frame.completed_tasks == frame.on_time_completed_tasks + frame.sla_violations).all(),name+' completed')
    np.testing.assert_allclose(frame.on_time_completion_rate,
                               frame.on_time_completed_tasks / frame.total_tasks,rtol=1e-12)
    require(((frame.on_time_completion_rate>=0)&(frame.on_time_completion_rate<=1)).all(),name+' rates')


def compare_file(relative, keys):
    actual = RESULTS / relative
    if not actual.is_file():
        raise FileNotFoundError(f'Rerun file missing: {actual}. Use python run.py all first.')
    saved = pd.read_csv(REFERENCE_RESULTS / relative)
    new = pd.read_csv(actual)
    require(not saved.duplicated(keys).any() and not new.duplicated(keys).any(), relative+' duplicate key')
    saved = saved.sort_values(keys).reset_index(drop=True)
    new = new.sort_values(keys).reset_index(drop=True)
    require(len(saved)==len(new),relative+' row count')
    require(set(saved.columns)==set(new.columns),relative+' columns')
    numeric = 0
    for col in saved.columns:
        a,b = saved[col],new[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            np.testing.assert_allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                                       rtol=1e-10,atol=1e-8,equal_nan=True,
                                       err_msg=f'{relative}: {col}')
            numeric += 1
        else:
            require(a.fillna('').astype(str).equals(b.fillna('').astype(str)),f'{relative}: {col}')
    return dict(file=relative,rows=len(saved),numeric_columns=numeric,matched=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--compare',action='store_true',help='Also compare the complete regenerated results to the reference CSVs.')
    args=ap.parse_args()
    manifest = ROOT/'CHECKSUMS.sha256'
    file_checks=0
    if manifest.is_file():
        for line in manifest.read_text().splitlines():
            digest,relative=line.split('  ',1)
            path=ROOT/relative
            require(path.is_file(),'Missing file: '+relative)
            require(hashlib.sha256(path.read_bytes()).hexdigest()==digest,'Checksum mismatch: '+relative)
            file_checks+=1
    workloads=pd.read_csv(DATA/'synthetic/workloads.csv')
    require(len(workloads)==150,'Expected 150 distinct synthetic workloads')
    for row in workloads.itertuples(index=False):
        require(hashlib.sha256((DATA/'synthetic'/row.path).read_bytes()).hexdigest()==row.file_sha256,row.path)
    main_runs=pd.read_csv(REFERENCE_RESULTS/'main/runs.csv')
    group=main_runs.groupby(['scenario','run_id','network_model','bandwidth_mbps'])
    require(len(main_runs)==3600 and len(group)==450,'Main comparison size')
    require(group['mode'].nunique().eq(8).all(),'Every main block must include eight policies')
    require(group.workload_sha256.nunique().eq(1).all(),'Unmatched main workloads')
    require(group.environment_sha256.nunique().eq(1).all(),'Unmatched main environments')
    check_accounting(main_runs,'main')
    mapping=pd.read_csv(REFERENCE_RESULTS/'google/mapping_sensitivity_20k.csv')
    group=mapping.groupby(['anchor','mapping','network_model','bandwidth_mbps'])
    require(len(mapping)==720 and len(group)==90,'Google mapping size')
    require(mapping.total_tasks.eq(20000).all(),'Google mapping task scale')
    require(group.workload_sha256.nunique().eq(1).all(),'Unmatched mapping workloads')
    require(group.environment_sha256.nunique().eq(1).all(),'Unmatched mapping environments')
    check_accounting(mapping,'mapping')
    coeff=pd.read_csv(REFERENCE_RESULTS/'google/coefficient_sensitivity_20k.csv')
    require(len(coeff)==1248 and coeff.coefficient_case.nunique()==26,'Coefficient size')
    check_accounting(coeff,'coefficients')
    tests=pd.read_csv(REFERENCE_RESULTS/'tables/secondary_four_policy_tests.csv')
    require(len(tests)==40 and tests.k_policies.eq(4).all() and tests.n_blocks.eq(30).all(),'Table VIII must use four policies')
    require(int((tests.rm_gg_holm_40<.05).sum())==39,'Table VIII ANOVA summary')
    require(int((tests.friedman_holm_40<.05).sum())==39,'Table VIII Friedman summary')
    native=pd.read_csv(DATA/'google/selected_raw_submit_events.csv')
    require(len(native)==20000 and int(native['time'].min())>=600000000,'Google extraction window')
    spatial=pd.read_csv(REFERENCE_RESULTS/'locality/run_results.csv')
    require(len(spatial)==3360,'Locality row count')
    require((spatial.tasks==spatial.on_time+spatial.late+spatial.blocked).all(),'Locality accounting')
    require(spatial.groupby(['run','bandwidth_mbps','topology','radius_m']).owner_sha256.nunique().eq(1).all(),'Locality assignment matching')
    analytical=pd.read_csv(REFERENCE_RESULTS/'analytical/mmck_runs.csv')
    require(len(analytical)==90,'Analytical replication count')
    report=dict(checksummed_files=file_checks,synthetic_inputs_checked=150,
                main_runs=3600,matched_main_conditions=450,google_mapping_evaluations=720,
                google_coefficient_evaluations=1248,table_VIII_comparisons=40,table_VIII_policies=4,
                locality_evaluations=3360,mmck_replications=90,all_checks_passed=True)
    if args.compare:
        report['regenerated_files']=[compare_file(relative,keys) for relative,keys in COMPARISONS.items()]
    folder=RESULTS/'verification';folder.mkdir(parents=True,exist_ok=True)
    (folder/('rerun_comparison.json' if args.compare else 'reference_check.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
