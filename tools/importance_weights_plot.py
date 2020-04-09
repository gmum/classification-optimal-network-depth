import argparse
import re
from pathlib import Path

import numpy as np
from matplotlib import colors, cm
from matplotlib import pyplot as plt
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import axes3d
from scipy.special import softmax
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record

# seaborn.set()
DEF_SIZE = 24
plt.rc('font', size=DEF_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=DEF_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=DEF_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=DEF_SIZE - 4)  # fontsize of the tick labels
plt.rc('ytick', labelsize=DEF_SIZE - 4)  # fontsize of the tick labels
plt.rc('legend', fontsize=DEF_SIZE - 8)  # legend fontsize
plt.rc('figure', titlesize=DEF_SIZE)  # fontsize of the figure title
EVENTS_DIR_REGEX = re.compile(r'.*?_args_.*(\d+)(?!@state)')
WEIGHTS_DIR_REGES = re.compile(r'Weights_(\d+)')


def my_summary_iterator(path):
    for r in tf_record.tf_record_iterator(str(path)):
        yield event_pb2.Event.FromString(r)


def plot_3d(weights, steps, timescale_name, max_ind, num_steps, plot_path):
    SMALL_SIZE = 18
    MEDIUM_SIZE = 22
    BIGGER_SIZE = 26
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # print(X.shape)
    # print(Y.shape)
    # print(weights.shape)
    # for plt_type in ['wireframe', 'surface', 'bar', 'bar3d']:
    xs = np.arange(0, max_ind + 1)
    ys = np.arange(0, num_steps)
    X, Y = np.meshgrid(xs, ys)
    x_raveled, y_raveled = X.ravel(), Y.ravel()

    for plt_type in ['bar3d']:
        fig = plt.figure(figsize=(16, 9))
        ax = plt.axes(projection='3d')
        ax.view_init(elev=45., azim=-75.)
        top = weights.ravel()
        bottom = np.zeros_like(top)
        width = 0.35
        depth = 1 / num_steps * 100
        if plt_type == 'wireframe':
            ax.plot_wireframe(X, Y, weights)
        elif plt_type == 'surface':
            ax.plot_surface(X, Y, weights, cmap=plt.cm.viridis)
        elif plt_type == 'bar':
            ax.bar(x_raveled, top, y_raveled)
        elif plt_type == 'bar3d':
            ax.set_zlim(0.0, 1.0)
            ax.bar3d(x_raveled, y_raveled, bottom, width, depth, top, shade=True, zsort='max')
        xs_readable = np.linspace(0, max_ind - 1, 10, dtype=np.int32)
        ax.set_xticks(xs_readable)
        ax.set_xticklabels([str(x + 1) for x in xs_readable])
        ax.set_xlabel('\n\nLayer')
        ys_readable = np.linspace(0, num_steps - 1, 8, dtype=np.int32)
        ax.set_yticks(ys_readable)
        ax.set_yticklabels([str(steps[y]) for y in ys_readable])
        ax.set_ylabel(f'\n\n{timescale_name}')
        ax.set_zlabel('\nWeight')
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f'Saving to {plot_path}')
        plt.close(fig)


def plot_brightness(weights, steps, timescale_name, plot_path, cmap_name="grey_r", scale="linear", colorbar=True):
    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()
    plt.xlabel(f"{timescale_name}")
    plt.ylabel("Layer")

    cmap = plt.get_cmap(cmap_name)
    if scale == "log":
        log_weights = np.log(weights.T)
        log_weights = log_weights - log_weights.min()
        log_weights /= log_weights.max()
        inputs = log_weights
        norm = colors.LogNorm(vmin=1e-8, vmax=1.)
        sm = cm.ScalarMappable(norm, cmap)
    elif scale == "linear":
        inputs = weights.T
        sm = cm.ScalarMappable(colors.Normalize(), cmap)
    elif scale == "linear_scaled":
        inputs = weights.T
        inputs /= inputs.max(0)
        sm = cm.ScalarMappable(colors.Normalize(), cmap)

    rgb_vals = cmap(inputs)

    # im = ax.imshow(rgb_vals, aspect=20)
    im = ax.imshow(rgb_vals, aspect='auto')
    max_ind = weights.shape[1]
    ys_step = round((max_ind - 1) / 10)
    ys_readable = range(0, max_ind - 1, ys_step)
    plt.yticks(ys_readable, [str(y + 1) for y in ys_readable])
    plt.ylim(-0.5, max_ind - 0.5)


    xs_readable = np.linspace(0, len(steps) - 1, 8, dtype=np.int32)
    plt.xticks(xs_readable, list(steps[i] for i in xs_readable))

    if colorbar:
        plt.colorbar(sm)
    corrected_plot_path = str(plot_path).replace(".png", f"_{scale}_{cmap_name}.png")
    fig.savefig(corrected_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f'Saving to {corrected_plot_path}')


def plot_surface(weights, plot_path):
    print(weights.shape)
    fig = plt.figure(figsize=(20, 10))
    ax = plt.gca()
    im = ax.stackplot(np.arange(weights.shape[0]), weights.T, labels=range(1, 20))
    plt.legend()
    plt.xlabel("Batch")
    # plt.ylabel("Weight")
    plt.tight_layout()

    # plt.show()
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size=1, pad=0)
    plt.show(fig)
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f'Saving to {plot_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help='Directory containing the results.', type=Path)
    parser.add_argument('--batches_per_epoch', type=int)
    parser.add_argument('--method', help='Method of plotting the weights', choices=["3d", "brightness", "surface"],
                        default="3d")
    parser.add_argument('--shell', help='Spawn IPython shell after completion', action='store_true')
    args = parser.parse_args()

    for child in args.dir.iterdir():
        if 'baseline' in child.name:
            continue
        match = EVENTS_DIR_REGEX.match(str(child))
        if match and child.is_dir():
            # get the number of weights
            max_ind = 0
            for subchild in child.iterdir():
                submatch = WEIGHTS_DIR_REGES.match(subchild.name)
                if submatch:
                    max_ind = max(max_ind, int(submatch.group(1)))
            # and the number of samples
            saved_xs = set()
            if not (child / f'Weights_{max_ind}').is_dir():
                print(f'Missing weight {max_ind}, skipping')
                continue
            for events_file in (child / f'Weights_{max_ind}').iterdir():
                for event in my_summary_iterator(events_file):
                    saved_xs.add(event.step)
            steps_list = sorted(saved_xs)
            num_steps = len(steps_list)
            steps = {step: i for i, step in enumerate(steps_list)}
            weights = np.zeros((num_steps, max_ind + 1))
            for j in range(max_ind + 1):
                for events_file in (child / f'Weights_{j}').iterdir():
                    for event in my_summary_iterator(events_file):
                        index = steps[event.step]
                        for value in event.summary.value:
                            weights[index][j] = value.simple_value
            for i in range(num_steps):
                weights[i] = softmax(weights[i])

            if args.batches_per_epoch:
                steps_reversed = {i: int(step / args.batches_per_epoch) for step, i in steps.items()}
                timescale_name = 'Epochs'
            else:
                steps_reversed = {i: step for step, i in steps.items()}
                timescale_name = 'Batches'

            plot_filename = f'{child.name}_{args.method}_plot.png'
            plot_path = args.dir / plot_filename
            if args.method == "3d":
                plot_3d(weights, steps_reversed, timescale_name, max_ind, num_steps, plot_path)
            elif args.method == "brightness":
                plot_brightness(weights, steps_reversed, timescale_name, plot_path, cmap_name="summer", scale="log",
                                colorbar=True)
            elif args.method == "surface":
                plot_surface(weights, plot_path)


if __name__ == '__main__':
    main()
