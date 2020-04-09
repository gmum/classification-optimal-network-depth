import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import axes3d

seaborn.set()
DEF_SIZE = 26
plt.rc('font', size=DEF_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=DEF_SIZE + 4)  # fontsize of the axes title
plt.rc('axes', labelsize=DEF_SIZE + 2)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=DEF_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=DEF_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=DEF_SIZE)  # legend fontsize
plt.rc('figure', titlesize=DEF_SIZE + 8)  # fontsize of the figure title

EVENTS_DIR_REGEX = re.compile(r'.*?_args_.*(\d+)(?!@state)')
WEIGHTS_DIR_REGES = re.compile(r'Weights_(\d+)')


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('Graph choice', help='graph choice csv data', type=Path)
    parser.add_argument('left_graph', help='left graph baseline csv data', type=Path)
    parser.add_argument('right_graph', help='right graph baseline csv data', type=Path)
    parser.add_argument('--batches_per_epoch', type=int)
    parser.add_argument('--shell', help='Spawn IPython shell after completion', action='store_true')
    args = parser.parse_args()

    df_left = pd.read_csv(args.left_graph, sep=',')
    df_right = pd.read_csv(args.right_graph, sep=',')
    fig, ax = plt.subplots(figsize=(16, 9))
    # ax.set_xticks(xs)
    # ax.set_xticklabels([str(x) for x in xs])
    ax.plot(df_left['Step'], df_left['Value'], label='ER graph')
    ax.plot(df_right['Step'], df_right['Value'], label='Random DAG')
    ax.set_xlabel('Batch' if not args.batches_per_epoch else 'Epoch')
    ax.set_ylabel('Train loss')
    ax.legend()
    plt.tight_layout()
    save_path = Path(f'graph_choice_loss.png')
    plt.savefig(save_path)
    print(f'Saving to: {save_path.resolve()}')

if __name__ == '__main__':
    main()
