from pathlib import Path
from run_simulation import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, load_inputs, run_sensitivity_analysis, aggregate_with_ci

def main() -> None:
    input_dir = DEFAULT_INPUT_DIR
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks_df, params_df = load_inputs(input_dir)
    print('Running sensitivity analysis only...')
    print(f'Input folder: {input_dir.resolve()}')
    print(f'Output folder: {output_dir.resolve()}')
    print(f'Total workload records: {len(tasks_df):,}')
    sensitivity_df = run_sensitivity_analysis(tasks_df, params_df)
    sensitivity_df.to_csv(output_dir / 'sensitivity_results_30runs.csv', index=False)
    sensitivity_summary_df = aggregate_with_ci(sensitivity_df, ['experiment_type', 'parameter_value', 'scenario', 'mode'])
    sensitivity_summary_df.to_csv(output_dir / 'sensitivity_summary_with_ci.csv', index=False)
    print('\nDone.')
    print('Created:')
    print(output_dir / 'sensitivity_results_30runs.csv')
    print(output_dir / 'sensitivity_summary_with_ci.csv')
if __name__ == '__main__':
    main()
