import argparse
import re
from itertools import cycle
from operator import itemgetter
from pathlib import Path

import numpy as np
import seaborn
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

seaborn.set()
DEF_SIZE = 24
plt.rc('font', size=DEF_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=DEF_SIZE + 4)  # fontsize of the axes title
plt.rc('axes', labelsize=DEF_SIZE + 2)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=DEF_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=DEF_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=DEF_SIZE)  # legend fontsize
plt.rc('figure', titlesize=DEF_SIZE + 8)  # fontsize of the figure title

BASELINE_STATE_FILE_REGEX = re.compile(r'.*_args_\(\d+, \d+, (\d+).*?_baseline_.*?(\d+)@state')
GRAPHS_BASELINE_STATE_FILE_REGEX = re.compile(r'.*?_baseline_(\d+)_(\d+)@state')
MODEL_ARGS_REGEX = re.compile(r'.*?_args_\((.*?)\).*?(\d+)@state')
GRAPH_MODEL_ARGS_REGEX = re.compile(r'.*?_beta_(.*?)_.*?_(\d+)@state')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help='Directory containing the results.', type=Path)
    parser.add_argument('--loss', help='Plot loss instead of accuracy', action='store_true')
    parser.add_argument('--no_legend', help='Omit legend', action='store_true')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--add', help='Additional runs to plot', nargs='+', type=Path)
    group.add_argument('--manually', help='File', type=Path)
    parser.add_argument('--show-single-example', help='Show only one run for our method instead of all the runs',
                        action='store_true')
    parser.add_argument('--shell', help='Spawn IPython shell after completion', action='store_true')
    args = parser.parse_args()

    postfix = 'loss' if args.loss else 'acc'
    filename_postfix = '_losses' if args.loss else ''

    results_baselines = {}
    train_results_baselines = {}
    for child in args.dir.iterdir():
        regex = GRAPHS_BASELINE_STATE_FILE_REGEX if 'Graph' in child.name else BASELINE_STATE_FILE_REGEX
        match = regex.match(str(child))
        if match:
            with open(child, 'rb') as f:
                res_dict = torch.load(f, map_location='cpu')
            key = int(match.group(1))
            score = res_dict[f'final_{postfix}']
            train_score = res_dict[f'final_train_{postfix}']
            results_baselines.setdefault(key, []).append(score)
            train_results_baselines.setdefault(key, []).append(train_score)
            print(f'Appending score {score} to key {key}.')
            print(f'Appending train score {train_score} to key {key}.')
        else:
            print(f'{child.name} does not match "{regex.pattern}"')

    results = []
    if args.add:
        for state_path in args.add:
            assert state_path.exists() and not state_path.is_dir()
            with open(state_path, 'rb') as f:
                res_dict = torch.load(f, map_location='cpu')
            regex = GRAPH_MODEL_ARGS_REGEX if 'Graph' in state_path.name else MODEL_ARGS_REGEX
            args_match = regex.match(state_path.name)
            # TODO handle different cases
            # this is messy - arguments aren't saved in any accessible form
            if 'FCNet' in state_path.name:
                args_str = args_match[1]
                model_args = args_str.split(sep=',')
                beta = model_args[5].strip()
            elif 'Graph' in state_path.name:
                beta = args_match[1].strip()
            else:
                args_str = args_match[1]
                model_args = args_str.split(sep=',')
                beta = model_args[6].strip()
            if f'cutout_final_{postfix}' in res_dict:
                chosen_layer = res_dict['chosen_layer'] + 1
                score = res_dict[f'cutout_final_{postfix}']
                train_score = res_dict[f'cutout_final_train_{postfix}']
                label = f'beta={beta}' if float(beta) > 0.0 else None
                results.append((chosen_layer, score, train_score, label))
            else:
                print(f'No cutout model results found in {state_path}')
    elif args.manually:
        with open(args.manually, 'rb') as f:
            for line in f:
                splt = line.split()
                results.append((int(splt[0]), float(splt[1]), float(splt[2]), float(splt[3])))
    print(f'Additional data: {results}')

    xs = np.array(sorted(results_baselines.keys()))
    values = np.array(list(results_baselines[k] for k in sorted(results_baselines.keys())))
    train_values = np.array(list(train_results_baselines[k] for k in sorted(results_baselines.keys())))
    # sample mean
    ys = values.mean(axis=1)
    train_ys = train_values.mean(axis=1)
    # sample standard deviation with Bessel's correction
    errs = values.std(axis=1, ddof=1)
    train_errs = train_values.std(axis=1, ddof=1)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    current_palette = cycle(seaborn.color_palette())
    current_color = next(current_palette)
    ax.errorbar(xs, ys, errs, color=current_color, fmt='-o',
                markersize=12, mew=2., mec="k",
                lw=2.5, capsize=5, capthick=3,
                )
    ax.errorbar(xs, train_ys, train_errs, color=current_color, fmt='-X',
                markersize=12, mew=2., mec="k",
                lw=2.5, capsize=5, capthick=3)
    handles = []
    handles += [Line2D([], [], color="k", marker="X", linestyle="None",
                       markersize=10, mew=0., mec="k", lw=2.5,
                       label="Train acc.")]
    handles += [Line2D([], [], color="k", marker="o", linestyle="None",
                       markersize=10, mew=0., mec="k", lw=2.5,
                       label="Test acc.")]
    handles += [Line2D([], [], color=current_color, marker="s", linestyle="None",
                       markersize=10, mew=0, mec="k",
                       label="Single-head classifiers")]
    # additional data
    results_cache = {}
    for r in results:
        # label = None if not r[3] else r[3]
        label = 'beta=0' if not r[3] else r[3]
        res = results_cache.setdefault(label, {'xs': [], 'train': [], 'test': []})
        res['xs'].append(r[0])
        res['train'].append(r[2])
        res['test'].append(r[1])
    for label, v in sorted(results_cache.items(), key=itemgetter(0)):
        c = next(current_palette)

        if args.show_single_example:
            x = np.bincount(v['xs']).argmax()
            chosen_indices = np.where(np.array(v['xs']) == x)[0]
            chosen_test = np.array(list(v['test'][idx] for idx in chosen_indices))
            chosen_train = np.array(list(v['train'][idx] for idx in chosen_indices))
            print(chosen_test, chosen_test.std())

            ax.errorbar(x, chosen_test.mean(), yerr=chosen_test.std(),
                        marker='o', linestyle="None", color=c, mfc=c,
                        markersize=17, mew=1.5, mec="k", zorder=10)
            fig_label = f'Our method, {label}'
            ax.errorbar(x, chosen_train.mean(), yerr=chosen_train.std(),
                        marker='X', linestyle="None", color=c, mfc=c,  # label=label_train,
                        markersize=17, mew=1.5, mec="k", zorder=10)
            handles += [Line2D([], [], linestyle="None",
                               marker='s', color=c, mfc=c, label=fig_label,
                               markersize=10, mew=0, mec="k", zorder=10)]
        else:
            ax.plot(v['xs'], v['test'],
                    marker='o', linestyle="None", color=c, mfc=c,
                    markersize=17, mew=1.5, mec="k", zorder=10)
            fig_label = f'Our method, {label}'
            ax.plot(v['xs'], v['train'],
                    marker='X', linestyle="None", color=c, mfc=c,  # label=label_train,
                    markersize=17, mew=1.5, mec="k", zorder=10)
            handles += [Line2D([], [], linestyle="None",
                               marker='s', color=c, mfc=c, label=fig_label,
                               markersize=10, mew=0, mec="k", zorder=10)]

    plt.rcParams.update({'lines.markeredgewidth': 1})
    # ax.bar(xs, ys, yerr=errs, align='center', alpha=0.7, color='blue', error_kw=dict(lw=5, capsize=5, capthick=3))
    # ax.bar(xs, train_ys, yerr=train_errs, align='center', alpha=0.3, color='orange',
    #        error_kw=dict(lw=5, capsize=5, capthick=3))
    if 'raph' in args.dir.name:
        ax.set_xlabel('Node index')
    else:
        ax.set_xlabel('Number of layers')
    # ax.set_ylabel('Accuracy')
    # ax.set_ylabel('Loss')
    if not args.no_legend:
        ax.legend(handles=handles)
    plt.tight_layout()
    save_path = args.dir / f'{str(args.dir)}_baseline_bar_plot{filename_postfix}.png'
    plt.savefig(save_path, dpi=300)
    print(f'Saving to: {save_path}')


if __name__ == '__main__':
    main()
