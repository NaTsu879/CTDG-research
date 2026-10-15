"""
Per-readout ablation of CRAFTV8 (CRAFT + reciprocity, long-range memory and co-visit closure readouts).

Every variant uses the best configs (--load_best_configs), so the rows differ only in which readouts are on:
the full model, the model without each readout in turn, and the model with all three off, which is CRAFT.

    python run_ablation.py --datasets uci wikipedia          # train what is missing, then print the table
    python run_ablation.py --datasets uci wikipedia --report_only
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np

MODEL = 'CRAFTV8'
DATASETS = ['uci', 'wikipedia', 'lastfm', 'GoogleLocal']

# (row name, version tag, extra flags)
VARIANTS = [
    ('CRAFTV8 (full)', 'v8_full', []),
    ('w/o reciprocity', 'v8_norecip', ['--no_reciprocity']),
    ('w/o long-range memory', 'v8_nomemory', ['--no_history_memory']),
    ('w/o co-visit closure', 'v8_nocovisit', ['--no_covisit']),
    ('CRAFT (all readouts off)', 'v8_craft', ['--no_reciprocity', '--no_history_memory', '--no_covisit']),
]

METRICS = [('mrr', 'MRR'), ('average_precision', 'AP'), ('roc_auc', 'AUC')]


def result_files(dataset, version):
    return sorted(glob.glob(f'./saved_results/{dataset}/{MODEL}/{MODEL}_seed*_v{version}.json'))


def run(dataset, version, flags, args):
    if len(result_files(dataset, version)) >= args.num_runs:
        print(f'[skip] {dataset} {version}: results already saved')
        return
    cmd = [sys.executable, 'train_link_prediction.py', '--dataset_name', dataset, '--model_name', MODEL,
           '--load_best_configs', '--gpu', str(args.gpu), '--num_runs', str(args.num_runs),
           '--seed', str(args.seed), '--version', version] + flags
    os.makedirs('./logs/ablation', exist_ok=True)
    log_path = f'./logs/ablation/{dataset}_{version}.log'
    print(f'[run] {" ".join(cmd)}  > {log_path}')
    with open(log_path, 'w') as log:
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
    if code != 0:
        print(f'[fail] {dataset} {version} exited with {code}, see {log_path}')
    if args.delete_checkpoints:
        # the test metrics are already in saved_results; the checkpoints of this variant are no longer needed
        for folder in glob.glob(f'./saved_models/{dataset}/{MODEL}/{MODEL}_seed*_v{version}'):
            shutil.rmtree(folder, ignore_errors=True)


def collect(dataset, version):
    """mean and sample standard deviation over seeds of every metric, or None when nothing was saved"""
    runs = []
    for path in result_files(dataset, version):
        with open(path) as f:
            runs.append({k: float(v) for k, v in json.load(f)['test metrics'].items()})
    if not runs:
        return None
    ddof = 1 if len(runs) > 1 else 0
    return {m: (np.mean([r[m] for r in runs]), np.std([r[m] for r in runs], ddof=ddof), len(runs))
            for m, _ in METRICS if m in runs[0]}


def report(datasets):
    header = '| Variant | ' + ' | '.join(f'{d} {short}' for d in datasets for _, short in METRICS) + ' |'
    lines = [header, '|' + '---|' * (1 + len(datasets) * len(METRICS))]
    csv = ['variant,dataset,metric,mean,std,num_seeds']
    for name, version, _ in VARIANTS:
        cells = []
        for dataset in datasets:
            res = collect(dataset, version)
            for metric, _ in METRICS:
                if res is None or metric not in res:
                    cells.append('-')
                    continue
                mean, std, n = res[metric]
                cells.append(f'{100 * mean:.2f} ± {100 * std:.2f}')
                csv.append(f'{name},{dataset},{metric},{mean:.6f},{std:.6f},{n}')
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    table = '\n'.join(lines)
    print('\n' + table)
    os.makedirs('./saved_results/ablation', exist_ok=True)
    with open('./saved_results/ablation/craftv8_ablation.md', 'w', encoding='utf-8') as f:
        f.write(table + '\n')
    with open('./saved_results/ablation/craftv8_ablation.csv', 'w', encoding='utf-8') as f:
        f.write('\n'.join(csv) + '\n')
    print('\nsaved to ./saved_results/ablation/craftv8_ablation.md and craftv8_ablation.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_runs', type=int, default=3, help='seeds per variant')
    parser.add_argument('--seed', type=int, default=0, help='first seed')
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    parser.add_argument('--variants', nargs='+', default=[version for _, version, _ in VARIANTS],
                        help='version tags of the variants to run, e.g. v8_full v8_norecip')
    parser.add_argument('--delete_checkpoints', action='store_true',
                        help='delete the saved checkpoints of each variant once its results are saved, to save disk')
    parser.add_argument('--report_only', action='store_true')
    args = parser.parse_args()

    if not args.report_only:
        for dataset in args.datasets:
            for _, version, flags in VARIANTS:
                if version in args.variants:
                    run(dataset, version, flags, args)
    report(args.datasets)
