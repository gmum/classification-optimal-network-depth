import argparse
import datetime
import math
import signal
import traceback
from itertools import chain
from pathlib import Path

import IPython
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import close
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
from tqdm import tqdm

from common import test_classification, cutout_baseline
from utils import loader, get_device, load_or_run_n, matrix_to_figure, cs_vec_to_figure

# for archiving purposes
SAVED_SOURCE = Path(__file__).read_text()

MODEL_SAVE_MINUTES = 5
IMAGE_SAVE_COUNT = 100


def train_classifier(context, train_data, test_data, batch_size, batch_xs, Model, model_args, init, Criterion,
                     Optimizer, optimizer_args, importance_optimizer_args, w_init='uniform', calculate_cosines=False):
    model = Model(*model_args).to(get_device())
    optimizer = Optimizer(model.parameters(), *optimizer_args)
    if not model.baseline:
        importance_optimizer = Optimizer(model.importance_parameters(), *importance_optimizer_args)
    criterion = Criterion(reduction='mean')
    image_batch_xs = [x for i, x in enumerate(batch_xs) if i % (len(batch_xs) // IMAGE_SAVE_COUNT) == 0]
    try:
        # batches_per_epoch = int(math.ceil(len(train_data) / batch_size))
        if context['model_state'] is None:
            model.apply(init)
            model.init_layer_importances(w_init)
            current_x = 0
            context['current_x'] = 0
            context['model_state'] = None
            context['optimizer_state'] = None
            context['batch_size'] = batch_size
            context['train_data_len'] = len(train_data)
            context['code'] = SAVED_SOURCE
        else:
            model.load_state_dict(context['model_state'])
            optimizer.load_state_dict(context['optimizer_state'])
            current_x = context['current_x']
        result_file = Path(context['dir_name']) / f"{context['run_name']}@state"
        summary_writer = SummaryWriter(f"{context['dir_name']}/{context['run_name']}")
        # summary_writer.add_graph(model, torch.randn(()))
        train_loader = loader(train_data, batch_size)
        test_loader = loader(test_data, batch_size)
        model.train()
        model_saved = datetime.datetime.now()
        with tqdm(total=batch_xs[-1], initial=current_x, unit_scale=True, dynamic_ncols=True) as pbar:
            while current_x <= batch_xs[-1]:
                for X, y in train_loader:
                    X = X.to(get_device())
                    y = y.to(get_device())

                    # Diagnostics
                    if current_x in batch_xs:
                        now = datetime.datetime.now()
                        pbar.set_description(desc=f'Last save {(now - model_saved).total_seconds():.0f}s ago',
                                             refresh=False)
                        if calculate_cosines and not model.baseline and (
                                current_x in image_batch_xs or current_x == batch_xs[-1]):
                            y_pred, layer_outputs = model(X)
                            loss, pen_loss = model.calculate_loss(y_pred, y, criterion)
                            non_head_parameters = list(model.non_head_parameters())
                            assert len(non_head_parameters) == len(layer_outputs) == model.num_heads

                            # for every layer separately
                            for i in range(model.num_heads):
                                layer_gradients = []
                                # calculate gradients for every head
                                for j in range(model.num_heads):
                                    if j < i:
                                        layer_gradients.append(
                                            torch.cat([torch.zeros_like(p).view(-1) for p in non_head_parameters[i]]))
                                        continue
                                    # calculate same output again, but with detach
                                    layer_outputs_detached = [layer_output.detach() if k != j else layer_output for
                                                              k, layer_output in enumerate(layer_outputs)]
                                    # and its loss
                                    layer_loss, _ = model.calculate_loss(layer_outputs_detached, y, criterion)
                                    assert layer_loss.allclose(loss)
                                    layer_gradient = torch.autograd.grad(layer_loss, non_head_parameters[i],
                                                                         retain_graph=True)
                                    layer_gradients.append(
                                        torch.cat([gradient.view(-1) for gradient in layer_gradient]))
                                # calculate stats
                                with torch.no_grad():
                                    # calculate norms
                                    # norms = {f'{j}': torch.norm(layer_gradient.view(-1), p=2).item() for
                                    #          (j, layer_gradient) in enumerate(layer_gradients)}
                                    # summary_writer.add_scalars(f'Layer {i} norms', norms, global_step=current_x)
                                    # calculate cosine similarities and put them into a matrix
                                    cs_matrix = torch.zeros(model.num_heads, model.num_heads, device=get_device())
                                    for j in range(i, len(layer_gradients)):
                                        for k in range(j, len(layer_gradients)):
                                            cs_matrix[k][j] = torch.nn.functional.cosine_similarity(
                                                layer_gradients[j].view(-1),
                                                layer_gradients[k].view(-1),
                                                dim=0)
                                            cs_matrix[j][k] = cs_matrix[k][j]
                                    fig = matrix_to_figure(cs_matrix)
                                    summary_writer.add_figure(f'Layer {i} cosine similarities', fig,
                                                              global_step=current_x)
                                    close(fig)
                                del layer_gradients, layer_gradient

                            true_grad_cs = torch.zeros(model.num_heads, device=get_device())
                            for i in range(model.num_heads):
                                target_params = non_head_parameters[0]
                                layer_outputs_detached = [layer_output.detach() if k != i else layer_output for
                                                          k, layer_output in enumerate(layer_outputs)]
                                layer_loss, _ = model.calculate_loss(layer_outputs_detached, y, criterion)
                                layer_gradient = torch.autograd.grad(layer_loss, target_params, retain_graph=True)
                                layer_gradient = torch.cat([gradient.view(-1) for gradient in layer_gradient])
                                true_grad = torch.autograd.grad(loss, target_params, retain_graph=True)
                                true_grad = torch.cat([gradient.view(-1) for gradient in true_grad])
                                true_grad_cs[i] = torch.nn.functional.cosine_similarity(
                                    layer_gradient.view(-1),
                                    true_grad.view(-1),
                                    dim=0)
                                del layer_gradient, true_grad
                            fig = cs_vec_to_figure(true_grad_cs)
                            summary_writer.add_figure(f'True cosine similarities (base layer 1)', fig,
                                                      global_step=current_x)
                            close(fig)

                            true_grad_cs = torch.zeros(model.num_heads, device=get_device())
                            for i in range(model.num_heads):
                                target_params = list(chain.from_iterable(non_head_parameters))
                                layer_outputs_detached = [layer_output.detach() if k != i else layer_output for
                                                          k, layer_output in enumerate(layer_outputs)]
                                layer_loss, _ = model.calculate_loss(layer_outputs_detached, y, criterion)
                                layer_gradient = torch.autograd.grad(layer_loss, target_params, retain_graph=True,
                                                                     allow_unused=True)
                                to_cat = []
                                for gradient, param in zip(layer_gradient, target_params):
                                    if gradient is not None:
                                        to_cat.append(gradient.view(-1))
                                    else:
                                        to_cat.append(torch.zeros_like(param).view(-1))
                                layer_gradient = torch.cat(to_cat)
                                true_grad = torch.autograd.grad(loss, target_params, retain_graph=True)
                                to_cat = []
                                for gradient, param in zip(true_grad, target_params):
                                    if gradient is not None:
                                        to_cat.append(gradient.view(-1))
                                    else:
                                        to_cat.append(torch.zeros_like(param).view(-1))
                                true_grad = torch.cat(to_cat)
                                true_grad_cs[i] = torch.nn.functional.cosine_similarity(
                                    layer_gradient.view(-1),
                                    true_grad.view(-1),
                                    dim=0)
                                del layer_gradient, true_grad
                            fig = cs_vec_to_figure(true_grad_cs)
                            summary_writer.add_figure(f'True cosine similarities', fig, global_step=current_x)
                            close(fig)


                        test_loss, test_acc = test_classification(model, test_loader, criterion, batches=10)
                        train_loss, train_acc = test_classification(model, train_loader, criterion, batches=10)
                        summary_writer.add_scalar('Eval/Test loss', test_loss, global_step=current_x)
                        summary_writer.add_scalar('Eval/Test accuracy', test_acc, global_step=current_x)
                        summary_writer.add_scalar('Eval/Train loss', train_loss, global_step=current_x)
                        summary_writer.add_scalar('Eval/Train accuracy', train_acc, global_step=current_x)
                        # save weights
                        if not model.baseline:
                            y_pred, layer_outputs = model(X)
                            loss, pen_loss = model.calculate_loss(y_pred, y, criterion)
                            weights = {f'{i}': value.item() for i, value in enumerate(model.ws)}
                            summary_writer.add_scalars('Weights', weights, global_step=current_x)
                            if current_x in image_batch_xs or current_x == batch_xs[-1]:
                                fig, ax = plt.subplots(figsize=(16, 9))
                                indices = np.arange(1, model.ws.size(0) + 1)
                                values = torch.softmax(model.ws.detach().cpu(), dim=0).numpy()
                                ax.bar(indices, values)
                                summary_writer.add_figure('Weights', fig, global_step=current_x)
                                close(fig)

                                layer_weights_gradient = torch.autograd.grad(loss, model.ws)
                                fig, ax = plt.subplots(figsize=(16, 9))
                                indices = np.arange(1, model.ws.size(0) + 1)
                                values = layer_weights_gradient[0].cpu().numpy()
                                ax.bar(indices, values)
                                summary_writer.add_figure('Weights grad', fig, global_step=current_x)
                                close(fig)

                                layer_losses = [criterion(layer_output, y).item() for layer_output in layer_outputs]
                                fig, ax = plt.subplots(figsize=(16, 9))
                                indices = np.arange(1, model.ws.size(0) + 1)
                                values = layer_losses
                                ax.bar(indices, values)
                                summary_writer.add_figure('Layer losses', fig, global_step=current_x)
                                close(fig)
                        # save model conditionally
                        if (now - model_saved).total_seconds() > 60 * MODEL_SAVE_MINUTES:
                            # save training state
                            context['current_x'] = current_x
                            context['model_state'] = model.state_dict()
                            context['optimizer_state'] = optimizer.state_dict()
                            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
                            with open(result_file, 'wb') as f:
                                torch.save(context, f)
                            signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})
                            model_saved = datetime.datetime.now()

                    # Training step
                    y_pred, layer_outputs = model(X)

                    loss, pen_loss = model.calculate_loss(y_pred, y, criterion)
                    optimizer.zero_grad()
                    if not model.baseline:
                        importance_optimizer.zero_grad()
                    loss.backward(retain_graph=True if current_x in batch_xs else False)
                    optimizer.step()
                    if not model.baseline:
                        importance_optimizer.step()

                    summary_writer.add_scalar(f'Train/Loss', loss.item(), global_step=current_x)
                    summary_writer.add_scalar(f'Train/Penalization loss', pen_loss.item(), global_step=current_x)
                    summary_writer.add_scalar(f'Train/Sum of probabilities (exp)',
                                              y_pred.exp().mean().item(), global_step=current_x)

                    pbar.update()
                    if current_x >= batch_xs[-1]:
                        current_x += 1
                        break
                    else:
                        current_x += 1

        if 'final_acc' not in context:
            context['current_x'] = current_x
            context['model_state'] = model.state_dict()
            context['optimizer_state'] = optimizer.state_dict()
            test_loss, test_acc = test_classification(model, test_loader, criterion, batches=0)
            context['final_acc'] = test_acc
            context['final_loss'] = test_loss
            print(f'Final loss: {test_loss}')
            train_loss, train_acc = test_classification(model, train_loader, criterion, batches=0)
            context['final_train_acc'] = train_acc
            context['final_train_loss'] = train_loss
            print(f'Final train loss: {train_loss}')
            # save model to secondary storage
            signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
            with open(result_file, 'wb') as f:
                torch.save(context, f)
            signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})
        if not model.baseline:
            if 'cutout_final_acc' not in context:
                cut_model, chosen_layer, single_choice = cutout_baseline(model)
                test_loss, test_acc = test_classification(cut_model, test_loader, criterion, batches=0)
                train_loss, train_acc = test_classification(cut_model, train_loader, criterion, batches=0)
                context['chosen_layer'] = chosen_layer
                context['single_choice'] = single_choice
                context['cutout_final_acc'] = test_acc
                context['cutout_final_loss'] = test_loss
                context['cutout_final_train_acc'] = train_acc
                context['cutout_final_train_loss'] = train_loss
                # save model to secondary storage
                signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
                with open(result_file, 'wb') as f:
                    torch.save(context, f)
                signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})
            print(f"Final ACC (cutout model): {context['cutout_final_acc']}")
            print(f"Final train ACC (cutout model): {context['cutout_final_train_acc']}")
            print(f"Chosen layer (cutout model): {context['chosen_layer']}")
            print(f"Single choice (cutout model): {context['single_choice']}")
        print(f"Final ACC: {context['final_acc']}")
        print(f"Final train ACC: {context['final_train_acc']}")

    except KeyboardInterrupt as e:
        return context, e
    except Exception as e:
        context['exception'] = e
        context['traceback'] = traceback.format_exc()
    return context, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', help='Directory containing the datasets.', type=Path,
                        default=Path.home() / '.datasets')
    parser.add_argument('--shell', help='Spawn IPython shell after completion', action='store_true')
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    get_device()

    # dataset = 'mnist'
    dataset = 'cifar'

    # model = 'fc'
    # model = 'conv'
    model = 'resnet'
    # model = 'vgg'

    if model == 'fc':
        if dataset == 'mnist':
            from models import init_weights, FCNet
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ])
            train_data = datasets.MNIST(args.dataset_dir, train=True, download=True, transform=transform)
            test_data = datasets.MNIST(args.dataset_dir, train=False, download=True, transform=transform)
            Model = FCNet
            orig_model_args = [28 * 28, 1, 20, 200, 10]
            max_epoch = 40
            batch_size = 128
        elif dataset == 'cifar':
            from models import init_weights, FCNet
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            train_data = datasets.CIFAR10(args.dataset_dir, train=True, download=True, transform=transform_train)
            test_data = datasets.CIFAR10(args.dataset_dir, train=False, download=True, transform=transform_test)
            Model = FCNet
            orig_model_args = [32 * 32, 3, 20, 1000, 10]
            max_epoch = 150
            batch_size = 128
    elif model == 'conv':
        if dataset == 'mnist':
            from models import init_weights, DCNet
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ])
            train_data = datasets.MNIST(args.dataset_dir, train=True, download=True, transform=transform)
            test_data = datasets.MNIST(args.dataset_dir, train=False, download=True, transform=transform)
            Model = DCNet
            orig_model_args = [28, 1, 15, 50, 5, 10]
            max_epoch = 50
            batch_size = 128
        elif dataset == 'cifar':
            from models import init_weights, DCNet
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            train_data = datasets.CIFAR10(args.dataset_dir, train=True, download=True, transform=transform_train)
            test_data = datasets.CIFAR10(args.dataset_dir, train=False, download=True, transform=transform_test)
            Model = DCNet
            orig_model_args = [32, 3, 20, 50, 5, 10]
            max_epoch = 150
            batch_size = 128
    elif model == 'resnet':
        if dataset == 'cifar':
            from models import init_weights, ResNet, BasicBlock
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            train_data = datasets.CIFAR10(args.dataset_dir, train=True, download=True, transform=transform_train)
            test_data = datasets.CIFAR10(args.dataset_dir, train=False, download=True, transform=transform_test)
            Model = ResNet
            orig_model_args = [BasicBlock, [18, 18, 18], 10]
            max_epoch = 150
            batch_size = 128
        else:
            raise ValueError('TODO implement more datasets')
    elif model == 'vgg':
        from models import VGG
        init_weights = VGG.init_weights
        Model = VGG
        if dataset == 'mnist':
            transform = transforms.Compose([
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ])
            train_data = datasets.MNIST(args.dataset_dir, train=True, download=True, transform=transform)
            test_data = datasets.MNIST(args.dataset_dir, train=False, download=True, transform=transform)
            orig_model_args = [10]
            max_epoch = 50
            batch_size = 128
        if dataset == 'cifar':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            train_data = datasets.CIFAR10(args.dataset_dir, train=True, download=True, transform=transform_train)
            test_data = datasets.CIFAR10(args.dataset_dir, train=False, download=True, transform=transform_test)
            orig_model_args = [10]
            max_epoch = 150
            batch_size = 128

    Criterion = torch.nn.NLLLoss

    batches = math.ceil(len(train_data) / batch_size)
    max_batch = max_epoch * batches
    xs = [round(x) for x in np.linspace(0, max_batch - 1, num=600).tolist()]
    print(xs)

    # Optimizer = torch.optim.SGD
    # parameters for resnet from: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
    # orig_optimizer_args = [1e-2, 0.9, 0.0, 5e-4]
    # orig_optimizer_args = [1e-2, 0.9, 0.0, 0.0]
    # orig_importance_optimizer_args = [1e-2, 0.9, 0.0, 0.0]
    Optimizer = torch.optim.Adam
    adam_betas = (0.9, 0.999)
    # parameters for resnet from: https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
    orig_optimizer_args = [1e-3, adam_betas]
    orig_importance_optimizer_args = [1e-3, adam_betas]

    # ===============
    # dir_name = f'n_runs_{dataset}_{model}_testrun'
    # initialization = 'uniform'
    # beta = 0.0
    # normalization = False
    # cosines = False
    # optimizer_args = tuple(orig_optimizer_args)
    # importance_optimizer_args = tuple(orig_importance_optimizer_args)
    # model_args = tuple(orig_model_args + [beta, normalization])
    # key = f'{Model.__name__}_args_{model_args}_init_{initialization}_' \
    #       f'{Optimizer.__name__}_args_{optimizer_args}_importance_{importance_optimizer_args}_bs_{batch_size}'
    # load_or_run_n(3, dir_name, key, train_classifier, train_data, test_data, batch_size, xs, Model, model_args,
    #               init_weights, Criterion, Optimizer, optimizer_args, importance_optimizer_args, initialization,
    #               cosines)
    # ===============

    # ===============
    # cosines
    # dir_name = f'n_runs_{dataset}_{model}_cosine_similarities_final'
    # cosines = True
    # normalization = False
    # initialization = 'uniform'
    # for beta in [0.0, 1e-3, 1e-2]:
    #     optimizer_args = tuple(orig_optimizer_args)
    #     importance_optimizer_args = tuple(orig_importance_optimizer_args)
    #     model_args = tuple(orig_model_args + [beta, normalization])
    #     key = f'{Model.__name__}_args_{model_args}_init_{initialization}_' \
    #           f'{Optimizer.__name__}_args_{optimizer_args}_importance_{importance_optimizer_args}_bs_{batch_size}'
    #     load_or_run_n(1, dir_name, key, train_classifier, train_data, test_data, batch_size, xs, Model, model_args,
    #                   init_weights, Criterion, Optimizer, optimizer_args, importance_optimizer_args, initialization,
    #                   cosines)
    # ===============

    # ===============
    # grid search
    # dir_name = f'n_runs_{dataset}_{model}_inits_and_betas'
    # dir_name = f'n_runs_{dataset}_{model}_inits_and_betas_2'
    # cosines = False
    # for normalization in [False, 1.0, 0.9]:
    #     for beta in [0.0, 1e-4, 1e-3, 1e-2, 1e-1]:
    #         for initialization in ['first', 'uniform', 'last']:
    #             optimizer_args = tuple(orig_optimizer_args)
    #             importance_optimizer_args = tuple(orig_importance_optimizer_args)
    #             model_args = tuple(orig_model_args + [beta, normalization])
    #             key = f'{Model.__name__}_args_{model_args}_init_{initialization}_' \
    #                   f'{Optimizer.__name__}_args_{optimizer_args}_importance_{importance_optimizer_args}_bs_{batch_size}'
    #             load_or_run_n(1, dir_name, key, train_classifier, train_data, test_data, batch_size, xs, Model,
    #                           model_args, init_weights, Criterion, Optimizer, optimizer_args, importance_optimizer_args,
    #                           initialization, cosines)
    # ===============

    # ===============
    # baselines
    # dir_name = f'n_runs_{dataset}_{model}_baselines'
    # dir_name = f'n_runs_{dataset}_{model}_baselines_5_and_betas'
    # initialization = 'uniform'
    # normalization = False
    # cosines = False
    # for l in range(20, 2, -1):
    #     optimizer_args = tuple(orig_optimizer_args)
    #     importance_optimizer_args = tuple(orig_importance_optimizer_args)
    #     model_args = orig_model_args + [0.0, False, True, True]
    #     model_args[2] = l
    #     model_args = tuple(model_args)
    #     key = f'{Model.__name__}_args_{model_args}_baseline_' \
    #           f'{Optimizer.__name__}_args_{optimizer_args}_bs_{batch_size}'
    #     load_or_run_n(3, dir_name, key, train_classifier, train_data, test_data, batch_size, xs, Model, model_args,
    #                   init_weights, Criterion, Optimizer, optimizer_args, importance_optimizer_args, initialization)

    # most basic method
    # initialization = 'uniform'
    # beta = 0.0
    # normalization = False
    # cosines = False
    # optimizer_args = tuple(orig_optimizer_args)
    # importance_optimizer_args = tuple(orig_importance_optimizer_args)
    # for beta in [0.0, 0.001, 0.01, 0.1]:
    #     model_args = tuple(orig_model_args + [beta, normalization])
    #     key = f'{Model.__name__}_args_{model_args}_init_{initialization}_' \
    #           f'{Optimizer.__name__}_args_{optimizer_args}_importance_{importance_optimizer_args}_bs_{batch_size}'
    #     load_or_run_n(5, dir_name, key, train_classifier, train_data, test_data, batch_size, xs, Model, model_args,
    #                   init_weights, Criterion, Optimizer, optimizer_args, importance_optimizer_args, initialization,
    #                   cosines)
    # ===============

    # ===============
    # baselines - no BN
    # dir_name = f'n_runs_{dataset}_{model}_baselines_no_bn'
    # dir_name = f'n_runs_{dataset}_{model}_baselines_no_bn_5'
    # initialization = 'uniform'
    # normalization = False
    # cosines = False
    # for l in range(20, 1, -1):
    #     optimizer_args = tuple(orig_optimizer_args)
    #     importance_optimizer_args = tuple(orig_importance_optimizer_args)
    #     model_args = orig_model_args + [0.0, normalization, False, True]
    #     model_args[2] = l
    #     model_args = tuple(model_args)
    #     key = f'{Model.__name__}_args_{model_args}_baseline_' \
    #           f'{Optimizer.__name__}_args_{optimizer_args}_bs_{batch_size}'
    #     load_or_run_n(3, dir_name, key, train_classifier, train_data, test_data, batch_size, xs, Model, model_args,
    #                   init_weights, Criterion, Optimizer, optimizer_args, importance_optimizer_args, initialization)

    # most basic method - no BN
    # beta = 0.0
    # normalization = False
    # optimizer_args = tuple(orig_optimizer_args)
    # importance_optimizer_args = tuple(orig_importance_optimizer_args)
    # model_args = tuple(orig_model_args + [beta, normalization, False])
    # key = f'{Model.__name__}_args_{model_args}_init_{initialization}_' \
    #       f'{Optimizer.__name__}_args_{optimizer_args}_importance_{importance_optimizer_args}_bs_{batch_size}'
    # load_or_run_n(5, dir_name, key, train_classifier, train_data, test_data, batch_size, xs, Model, model_args,
    #               init_weights, Criterion, Optimizer, optimizer_args, importance_optimizer_args, initialization,
    #               cosines)
    # ===============

    # ===============
    # baselines MNIST FC
    # dir_name = f'n_runs_{dataset}_{model}_baselines'
    # dir_name = f'n_runs_{dataset}_{model}_baselines_5'
    # initialization = 'uniform'
    # cosines = False
    # for l in range(20, 0, -1):
    #     optimizer_args = tuple(orig_optimizer_args)
    #     importance_optimizer_args = tuple(orig_importance_optimizer_args)
    #     model_args = orig_model_args + [0.0, False, True]
    #     model_args[2] = l
    #     model_args = tuple(model_args)
    #     key = f'{Model.__name__}_args_{model_args}_baseline_' \
    #           f'{Optimizer.__name__}_args_{optimizer_args}_bs_{batch_size}'
    #     load_or_run_n(3, dir_name, key, train_classifier, train_data, test_data, batch_size, xs, Model, model_args,
    #                   init_weights, Criterion, Optimizer, optimizer_args, importance_optimizer_args, initialization)
    #
    # # most basic method
    # beta = 0.0
    # normalization = False
    # optimizer_args = tuple(orig_optimizer_args)
    # importance_optimizer_args = tuple(orig_importance_optimizer_args)
    # model_args = tuple(orig_model_args + [beta, normalization])
    # key = f'{Model.__name__}_args_{model_args}_init_{initialization}_' \
    #       f'{Optimizer.__name__}_args_{optimizer_args}_importance_{importance_optimizer_args}_bs_{batch_size}'
    # load_or_run_n(5, dir_name, key, train_classifier, train_data, test_data, batch_size, xs, Model, model_args,
    #               init_weights, Criterion, Optimizer, optimizer_args, importance_optimizer_args, initialization,
    #               cosines)
    # ===============

    # ===============
    # resnet betas and baseline
    dir_name = f'n_runs_{dataset}_{model}_ver_2'
    normalization = False
    initialization = 'uniform'
    cosines = False
    # our method, with betas
    for beta in [4e-3, 8e-3, 0.0, 2e-3, 1e-2]:
        optimizer_args = tuple(orig_optimizer_args)
        importance_optimizer_args = tuple(orig_importance_optimizer_args)
        model_args = tuple(orig_model_args + [beta, normalization])
        key = f'{Model.__name__}_args_{model_args}_init_{initialization}_' \
              f'{Optimizer.__name__}_args_{optimizer_args}_importance_{importance_optimizer_args}_bs_{batch_size}'
        load_or_run_n(1, dir_name, key, train_classifier, train_data, test_data, batch_size, xs, Model, model_args,
                      init_weights, Criterion, Optimizer, optimizer_args, importance_optimizer_args, initialization,
                      cosines)
    # original baseline
    optimizer_args = tuple(orig_optimizer_args)
    importance_optimizer_args = tuple(orig_importance_optimizer_args)
    model_args = tuple(orig_model_args + [0.0, normalization, True])
    key = f'{Model.__name__}_args_{model_args}_init_{initialization}_' \
          f'{Optimizer.__name__}_args_{optimizer_args}_importance_{importance_optimizer_args}_bs_{batch_size}'
    load_or_run_n(2, dir_name, key, train_classifier, train_data, test_data, batch_size, xs, Model, model_args,
                  init_weights, Criterion, Optimizer, optimizer_args, importance_optimizer_args, initialization,
                  cosines)
    # baseline every layer
    # TODO
    # ===============

    if args.shell:
        IPython.embed()


if __name__ == '__main__':
    main()
