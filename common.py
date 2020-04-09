from copy import deepcopy
from time import perf_counter

import numpy as np
import torch

from utils import get_device


def copy(v_iter):
    return [p.detach().clone() for p in v_iter]


def copy_(v1_iter, v2_iter):
    for p1, p2 in zip(v1_iter, v2_iter):
        p1.data.copy_(p2.data)


def squared_euclidean_distances(X, Y):
    # possibly inefficient
    m1 = X.unsqueeze(1).repeat(1, Y.size(0), 1)
    m2 = Y.unsqueeze(0).repeat(X.size(0), 1, 1)
    return (m1 - m2).pow(2).sum(2)


def abs_normalize(x, dim):
    y = (x).abs()
    return y / y.sum(dim=dim, keepdim=True)


def exp_normalize(x, dim):
    b = x.max(dim=dim, keepdim=True)[0]
    y = (x - b).exp()
    return y / y.sum(dim=dim, keepdim=True)


def test_classification(model, data_loader, criterion, batches=0, device=get_device(), eval=True):
    if eval:
        model.eval()
    with torch.no_grad():
        running_loss = 0.0
        correct, total = 0, 0
        for batch, (X, y) in enumerate(data_loader):
            if batch > batches > 0:
                break
            X = X.to(device)
            y = y.to(device)
            y_pred, _ = model(X)
            loss, _ = model.calculate_loss(y_pred, y, criterion)
            running_loss += loss.item()
            y_pred_max = y_pred.argmax(dim=1)
            correct += (y_pred_max == y).sum().item()
            total += y.size(0)
    if eval:
        model.train()
    # loss, acc
    return running_loss / (batch + 1), correct / total


def test_reconstruction(model, data_loader, criterion, batches=0, device=get_device(), eval=True):
    if eval:
        model.eval()
    with torch.no_grad():
        running_loss = 0.0
        for batch, (X, _) in enumerate(data_loader):
            if batch > batches > 0:
                break
            X = X.to(device)
            X_pred = model(X)
            loss = criterion(X_pred, X)
            running_loss += loss.item()
    if eval:
        model.train()
    return running_loss / (batch + 1)


def test_running_time(model, data_loader, batches=1000, device=get_device(), eval=True):
    if eval:
        model.eval()

    if device != "cpu":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

    running_times = []
    with torch.no_grad():
        running_loss = 0.0
        correct, total = 0, 0
        for batch, (X, y) in enumerate(data_loader):
            if batch > batches > 0:
                break
            X = X.to(device)
            y = y.to(device)
            if device == "cuda:0":
                start.record()
                y_pred, _ = model(X)
                end.record()
                torch.cuda.synchronize()
                diff_ms = start.elapsed_time(end)
                running_times.append(diff_ms)
            else:
                start = perf_counter()
                y_pred, _ = model(X)
                end = perf_counter()
                diff_ms = (end - start) * 1000
                running_times.append(diff_ms)

    if eval:
        model.train()

    running_times = running_times[5:]
    avg_running_time = np.mean(running_times)
    std_running_time = np.std(running_times)
    # print(running_times)
    return avg_running_time, std_running_time


def cutout_baseline(model):
    if model.baseline:
        return deepcopy(model)
    else:
        from models import ResNet
        if isinstance(model, ResNet):
            cut_model = deepcopy(model)
            softmaxed_ws = torch.softmax(cut_model.ws, dim=0)
            chosen_layer = softmaxed_ws.argmax().item()
            single_choice = all(w == 0.0 for i, w in enumerate(softmaxed_ws) if i != chosen_layer)
            cut_model.ws = torch.zeros_like(cut_model.ws)
            cut_model.ws[chosen_layer] = 1000.0
            return cut_model, chosen_layer, single_choice
        else:
            cut_model = deepcopy(model)
            softmaxed_ws = torch.softmax(cut_model.ws, dim=0)
            chosen_layer = softmaxed_ws.argmax().item()
            single_choice = all(w == 0.0 for i, w in enumerate(softmaxed_ws) if i != chosen_layer)
            cut_model.layers = cut_model.layers[:chosen_layer + 1]
            if hasattr(cut_model, 'batchnorm') and cut_model.batchnorm:
                cut_model.bn_layers = cut_model.bn_layers[:chosen_layer + 1]
            cut_model.layers.append(cut_model.heads[chosen_layer])
            cut_model.heads = None
            cut_model.ws = None
            cut_model.layer_penalties = None
            cut_model.baseline = True
            return cut_model, chosen_layer, single_choice
