"""
Minimal ablation of MYMODEL (CRAFT + reciprocity readout) on uci, wikipedia and Flickr.

Both variants use the best configs (--load_best_configs), so the only difference between the rows is the
reciprocity readout. Switching it off reproduces CRAFT.

    python run_ablation.py                 # train everything, then print the table
    python run_ablation.py --report_only   # only collect the saved results and print the table
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

DATASETS = ['uci', 'wikipedia', 'Flickr']

# (row name, version tag, extra flags)
VARIANTS = [
    ('MYMODEL (CRAFT + reciprocity)', 'recip', []),
    ('CRAFT (w/o reciprocity)', 'craft', ['--no_reciprocity']),
]

METRICS = [('mrr', 'MRR'), ('average_precision', 'AP'), ('roc_auc', 'AUC')]


def result_files(dataset, version):
    return sorted(glob.glob(f'./saved_results/{dataset}/MYMODEL/MYMODEL_seed*_v{version}.json'))


def run(dataset, version, flags, args):
    if len(result_files(dataset, version)) >= args.num_runs:
        print(f'[skip] {dataset} {version}: results already saved')
        return
    cmd = [sys.executable, 'train_link_prediction.py', '--dataset_name', dataset, '--model_name', 'MYMODEL',
           '--load_best_configs', '--gpu', str(args.gpu), '--num_runs', str(args.num_runs),
           '--seed', str(args.seed), '--version', version] + flags
    os.makedirs('./logs/ablation', exist_ok=True)
    log_path = f'./logs/ablation/{dataset}_{version}.log'
    print(f'[run] {" ".join(cmd)}  > {log_path}')
    with open(log_path, 'w') as log:
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
    if code != 0:
        print(f'[fail] {dataset} {version} exited with {code}, see {log_path}')


def collect(dataset, version):
    """mean and std over seeds of every metric, or None when nothing was saved"""
    runs = []
    for path in result_files(dataset, version):
        with open(path) as f:
            runs.append({k: float(v) for k, v in json.load(f)['test metrics'].items()})
    if not runs:
        return None
    return {m: (np.mean([r[m] for r in runs]), np.std([r[m] for r in runs]), len(runs))
            for m, _ in METRICS if m in runs[0]}


def report():
    header = '| Variant | ' + ' | '.join(f'{d} {short}' for d in DATASETS for _, short in METRICS) + ' |'
    lines = [header, '|' + '---|' * (1 + len(DATASETS) * len(METRICS))]
    csv = ['variant,dataset,metric,mean,std,num_seeds']
    for name, version, _ in VARIANTS:
        cells = []
        for dataset in DATASETS:
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
    with open('./saved_results/ablation/ablation.md', 'w') as f:
        f.write(table + '\n')
    with open('./saved_results/ablation/ablation.csv', 'w') as f:
        f.write('\n'.join(csv) + '\n')
    print('\nsaved to ./saved_results/ablation/ablation.md and ablation.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_runs', type=int, default=3, help='seeds per variant')
    parser.add_argument('--seed', type=int, default=0, help='first seed')
    parser.add_argument('--datasets', nargs='+', default=DATASETS)
    parser.add_argument('--report_only', action='store_true')
    args = parser.parse_args()
    DATASETS = args.datasets

    if not args.report_only:
        for dataset in DATASETS:
            for _, version, flags in VARIANTS:
                run(dataset, version, flags, args)
    report()
