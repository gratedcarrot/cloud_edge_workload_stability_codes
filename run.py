#!/usr/bin/env python3
"""Run verification, experiments and figure generation for the cloud–edge study.

Run ``python run.py --help`` from any directory. Outputs are written separately
from the supplied reference results. This program never uploads files or changes
an online repository.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
COMMANDS = ('verify', 'test', 'smoke', 'figures', 'tables', 'generate', 'main',
            'sensitivity', 'trace', 'proxy', 'locality', 'analytical', 'compare', 'all')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=COMMANDS)
    parser.add_argument('--workers', type=int, default=4,
                        help='Parallel workload processes (default: 4).')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs',
                        help='Rerun output folder; bundled results are left untouched.')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    output = args.output.expanduser().resolve()
    if output == ROOT / 'results' or output == ROOT:
        parser.error('Choose a separate output folder to preserve the reference results.')
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CLOUD_EDGE_OUTPUT=str(output))
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[name] = '1'
    full = args.command in ('all', 'trace', 'proxy', 'locality', 'test', 'smoke')
    if full and not (env.get('CXX') or shutil.which('g++') or shutil.which('clang++')):
        parser.error('This command requires a C++17 compiler (g++/clang++) on PATH or CXX. '
                     'Verification, table exports and saved-result figures need only Python.')

    def run(script: str, *extra: str) -> None:
        print(f'\nRunning {script}', flush=True)
        subprocess.run([sys.executable, str(ROOT / 'scripts' / script), *extra],
                       cwd=ROOT, env=env, check=True)

    def tests() -> None:
        subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                       cwd=ROOT, env=env, check=True)
        run('check_backend.py')

    def stages(command: str) -> None:
        if command == 'verify':
            run('verify_results.py')
        elif command == 'test':
            tests()
        elif command == 'smoke':
            run('smoke_test.py')
        elif command == 'generate':
            run('generate_workloads.py')
        elif command == 'main':
            run('run_main.py', '--workers', str(args.workers))
            run('analyze_main.py')
        elif command == 'sensitivity':
            run('run_network_sensitivity.py', '--workers', str(args.workers))
            run('run_capacity_sensitivity.py', '--workers', str(args.workers))
        elif command == 'trace':
            run('run_google_mapping.py', '--workers', str(args.workers))
        elif command == 'proxy':
            run('run_google_coefficients.py', '--workers', str(args.workers))
            run('run_synthetic_coefficients.py', '--workers', str(args.workers))
        elif command == 'locality':
            run('run_locality.py', '--workers', str(args.workers))
        elif command == 'analytical':
            run('check_mmck.py')
            run('check_workloads.py')
        elif command == 'tables':
            run('summarize_results.py')
            run('compare_baselines.py')
            run('export_tables.py')
        elif command == 'figures':
            run('plot_framework.py')
            run('plot_figures.py')
        elif command == 'compare':
            run('verify_results.py', '--compare')
        elif command == 'all':
            for stage in ('verify', 'test', 'main', 'sensitivity', 'trace', 'proxy',
                          'locality', 'analytical', 'tables', 'figures', 'compare'):
                stages(stage)
    try:
        stages(args.command)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
    print(f'\nFinished. Generated files: {output}', flush=True)


if __name__ == '__main__':
    main()
