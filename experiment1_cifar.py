import click
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from cifar_models import get_model
from model_saving import save_model, save_statistics
from classification_utils import train
from quantized_layers import QConv2d, FrozenConv2d

BATCH_SIZE = 100
DATA_PATH = "./datasets"

if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)


def pretrain_model(model, train_loader, test_loader, device, step_epochs, n_steps, lr_scale):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_epochs, lr_scale)
    loss = nn.CrossEntropyLoss()

    train_stat, test_stat = train(model, optimizer, train_loader, test_loader, loss,
                                  n_epochs=step_epochs * n_steps, scheduler=scheduler,
                                  device=device, use_tqdm=True, print_results=True)

    return train_stat, test_stat


def ptq_model(model_name: str, model, in_bins: int, w_bins: int, symmetric_mode: bool,
              train_loader, device='cpu', mode='ada'):
    model = model.to(device)
    qmodel = get_model(model_name)
    qmodel.load_state_dict(model.state_dict())
    qmodel = qmodel.to(device)

    cmodel = get_model(model_name)
    cmodel.load_state_dict(model.state_dict())
    cmodel = cmodel.to(device)
    N_BLOCKS = 50

    cache = []
    for i, (X, _) in enumerate(train_loader):
        if i == N_BLOCKS:
            break
        cache.append(X.to(device))
        cmodel.cache(X.to(device))

    in_step = -1
    for i, layer_name in enumerate(qmodel.layer_iterator()):
        layer = getattr(cmodel, layer_name)
        corrected_in = None
        if i > 0:
            for X in cache:
                qmodel.cache(X)
            prev_layer = getattr(qmodel, layer_name)
            corrected_in = prev_layer.cached_in
        qlayer = QConv2d(layer, in_bins, w_bins, symmetric_mode, in_step, mode, corrected_in).to(device)
        in_step = qlayer.input_quant.scale * qlayer.weight_quant.scale
        setattr(qmodel, layer_name, qlayer)
        qmodel.drop_cache()
    del cmodel
    return qmodel


def gradual_freeze_model(model_name: str, qmodel, train_loader, test_loader, device):
    EP_FREEZE = 3
    fmodel = get_model(model_name, quantized=True).to(device)
    fmodel.load_state_dict(qmodel.state_dict())

    train_stat, test_stat = None, None

    for layer_name in qmodel.freeze_layer_iterator():
        flayer = FrozenConv2d(getattr(fmodel, layer_name)).to(device)
        setattr(fmodel, layer_name, flayer)
        optimizer = optim.SGD(fmodel.parameters(), lr=1e-5, momentum=0.9)

        loss = nn.CrossEntropyLoss()
        train_stat, test_stat = train(fmodel, optimizer, train_loader, test_loader, loss,
                                      n_epochs=EP_FREEZE, train_stat=train_stat, test_stat=test_stat,
                                      device=device, use_tqdm=True, print_results=True)
    return fmodel, train_stat, test_stat


@click.command()
@click.option('-m', '--model',
              type=click.Choice(['CNN6', 'CNN7', 'CNN8', 'CNN9', 'CNN10'], case_sensitive=False))
@click.option('-p', '--prefix', type=click.STRING, default='')
@click.option('--step_epochs', type=click.INT, default=50)
@click.option('--n_steps', type=click.INT, default=4)
@click.option('--lr_scale', type=click.FLOAT, default=0.4)
@click.option('--device', type=click.STRING, default='cuda')
@click.option('--n_exp_rep', type=click.INT, default=1)
def main(model, prefix, step_epochs, n_steps, lr_scale, device, n_exp_rep):
    transformer_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(9),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
    ])

    transformer_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    print('Downloading dataset')
    train_set = torchvision.datasets.CIFAR10(DATA_PATH, train=True, transform=transformer_train, download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_set = torchvision.datasets.CIFAR10(DATA_PATH, train=False, transform=transformer_test, download=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    print('Done!')

    network = get_model(model)

    device = torch.device(device)
    name = f'{prefix}_{model}'
    print(f'Starting training on {device}')

    pretrain_start = time.time()
    train_stat, test_stat = pretrain_model(network, train_loader, test_loader, device, step_epochs, n_steps, lr_scale)
    pretrain_end = time.time()
    print(f'Finisher pretraining')
    save_model(network, f'{name}_baseline', test_loader, pretrain_end - pretrain_start, device=device)
    save_statistics(f'{name}_baseline', train_stat, test_stat)
    print(f'Starting quantization')

    for j in range(n_exp_rep):
        def quantization(in_bins, weight_bins, symmetric=True):
            print(f'Quantizing {in_bins}x{weight_bins}')
            quant_start = time.time()
            quantized_model = ptq_model(model, network, in_bins, weight_bins, symmetric, train_loader, device=device)
            save_model(quantized_model, f'{name}_{in_bins}_{weight_bins}_ptq__{j + 1}', test_loader, device=device)
            frozen_model, train_stat, test_stat = gradual_freeze_model(
                model, quantized_model, train_loader, test_loader, device=device)
            quant_end = time.time()
            save_model(frozen_model, f'{name}_{in_bins}_{weight_bins}_quant__{j + 1}', test_loader,
                       quant_end - quant_start, device)
            save_statistics(f'{name}_{in_bins}_{weight_bins}_quant__{j + 1}', train_stat, test_stat)

        quantization(16, 16, False)
        quantization(256, 256, False)
        for weight_bins in [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 29, 31, 37, 43, 51, 63, 85, 127]:
            in_bins = (127 // (weight_bins // 2)) * 2 + 1
            quantization(in_bins, weight_bins, True)

    print('Finished!')


if __name__ == '__main__':
    main()