import multiprocessing
import os
import subprocess
import traceback
from itertools import product

import numpy as np
import seaborn
import torch
from matplotlib import pyplot as plt

seaborn.set()
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


def get_gpu_memory():
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,nounits,noheader'
        ])
    gpu_memory = [int(x) for x in result.decode().strip().split()]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map



LOADER_WORKERS = 4
# PIN_MEMORY = True
PIN_MEMORY = False

device = None


def get_device():
    global device
    if device is None:
        print(f'{multiprocessing.cpu_count()} CPUs')

        print(f'{torch.cuda.device_count()} GPUs')
        if torch.cuda.is_available():
            device = 'cuda:0'
            # torch.set_default_tensor_type(torch.cuda.FloatTensor)
            for k, v in get_gpu_memory().items():
                print(f'Device {k} memory: {v} MiB')
            torch.backends.cudnn.benchmark = True
        else:
            # torch.set_default_tensor_type(torch.FloatTensor)
            device = 'cpu'
        print(f'Using: {device}')
    return device


def loader(data, batch_size):
    return torch.utils.data.DataLoader(dataset=data, batch_size=batch_size,
                                       shuffle=True,
                                       pin_memory=PIN_MEMORY,
                                       num_workers=LOADER_WORKERS)


def load_or_run(dir_name, run_name, method, *args, **kwargs):
    os.makedirs(dir_name, exist_ok=True)
    filepath = os.path.join(dir_name, f'{run_name}@state')
    print(f'State file: {filepath}')
    loaded = False
    if os.path.isfile(filepath):
        try:
            with open(filepath, 'rb') as f:
                context = torch.load(f, map_location=get_device())
                loaded = True
        except Exception:
            print(f'Exception when loading {filepath}')
            traceback.print_exc()
    if not loaded:
        context = {}
        context['model_state'] = None
        context['run_name'] = run_name
        context['dir_name'] = dir_name
    # TODO maybe move arguments into context?
    context, ex = method(context, *args, **kwargs)
    if ex is not None:
        raise ex
    if 'exception' in context:
        print(context['traceback'])
    return context


def load_or_run_n(n, dir_name, run_name, method, *args, **kwargs):
    results = []
    for i in range(n):
        name = f'{run_name}_{i}'
        results.append(load_or_run(dir_name, name, method, *args, **kwargs))
    return results


def matrix_to_figure(matrix, xlabel="", ylabel=""):
    matrix = matrix.cpu().numpy()
    fig, ax = plt.subplots(figsize=(16, 16), facecolor='w', edgecolor='k')
    ax.imshow(matrix, cmap='Spectral_r', vmin=-1, vmax=1)
    # set x axis
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([str(i) for i in np.arange(matrix.shape[1])], fontsize=18)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel(xlabel)
    # set y axis
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([str(i) for i in np.arange(matrix.shape[0])], fontsize=18)
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()
    ax.set_ylabel(ylabel)
    # plot text
    for i, j in product(range(matrix.shape[0]), range(matrix.shape[1])):
        ax.text(j, i, f'{matrix[i, j]:4.2f}' if matrix[i, j] != 0 else '.', horizontalalignment='center', fontsize=14,
                verticalalignment='center', color='black')
    ax.autoscale()
    fig.set_tight_layout(True)
    return fig


def cs_vec_to_figure(cs_vec, xlabel=""):
    cs_vec = cs_vec.cpu().numpy()
    fig, ax = plt.subplots(figsize=(22, 2), facecolor='w', edgecolor='k')
    ax.imshow(cs_vec.reshape(1, -1), cmap='Spectral_r', vmin=-1, vmax=1)
    ax.set_xticks(np.arange(cs_vec.shape[0]))
    ax.set_xticklabels([str(i) for i in np.arange(cs_vec.shape[0])], fontsize=18)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    for idx in range(len(cs_vec)):
        ax.text(idx, 0, f'{cs_vec[idx]:4.2f}' if cs_vec[idx] != 0 else '.', horizontalalignment='center', fontsize=14,
                verticalalignment='center', color='black')
    ax.autoscale()
    fig.set_tight_layout(True)
    return fig
