import click
import copy
import time
from operator import attrgetter

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
from torchvision.models import resnet18, ResNet18_Weights

from classification_utils import train, test
from model_saving import save_model, save_statistics
from resnet_common import MyQResNet, resnet_transformer_test, resnet_transformer_train
from quantized_layers import QConv2d, FrozenConv2d


def resnet18_setter_iterator(freeze=False):
    def setter00(model, layer):
        model.conv1 = layer

    def setter01(model, layer):
        model.layer1[0].conv1 = layer

    def setter02(model, layer):
        model.layer1[0].conv2 = layer

    def setter03(model, layer):
        model.layer1[1].conv1 = layer

    def setter04(model, layer):
        model.layer1[1].conv2 = layer

    def setter05(model, layer):
        model.layer2[0].conv1 = layer

    def setter06(model, layer):
        model.layer2[0].conv2 = layer

    def setter07(model, layer):
        model.layer2[0].downsample = layer

    def setter08(model, layer):
        model.layer2[1].conv1 = layer

    def setter09(model, layer):
        model.layer2[1].conv2 = layer

    def setter10(model, layer):
        model.layer3[0].conv1 = layer

    def setter11(model, layer):
        model.layer3[0].conv2 = layer

    def setter12(model, layer):
        model.layer3[0].downsample = layer

    def setter13(model, layer):
        model.layer3[1].conv1 = layer

    def setter14(model, layer):
        model.layer3[1].conv2 = layer

    def setter15(model, layer):
        model.layer4[0].conv1 = layer

    def setter16(model, layer):
        model.layer4[0].conv2 = layer

    def setter17(model, layer):
        model.layer4[0].downsample = layer

    def setter18(model, layer):
        model.layer4[1].conv1 = layer

    def setter19(model, layer):
        model.layer4[1].conv2 = layer

    if freeze:
        yield setter00
    yield setter01
    yield setter02
    yield setter03
    yield setter04
    yield setter05
    yield setter06
    yield setter07
    yield setter08
    yield setter09
    yield setter10
    yield setter11
    yield setter12
    yield setter13
    yield setter14
    yield setter15
    yield setter16
    yield setter17
    yield setter18
    yield setter19


def resnet18_getter_iterator(freeze=False):
    if freeze:
        yield attrgetter('conv1')
    yield attrgetter('layer1.0.conv1')
    yield attrgetter('layer1.0.conv2')
    yield attrgetter('layer1.1.conv1')
    yield attrgetter('layer1.1.conv2')
    yield attrgetter('layer2.0.conv1')
    yield attrgetter('layer2.0.conv2')
    yield attrgetter('layer2.0.downsample')
    yield attrgetter('layer2.1.conv1')
    yield attrgetter('layer2.1.conv2')
    yield attrgetter('layer3.0.conv1')
    yield attrgetter('layer3.0.conv2')
    yield attrgetter('layer3.0.downsample')
    yield attrgetter('layer3.1.conv1')
    yield attrgetter('layer3.1.conv2')
    yield attrgetter('layer4.0.conv1')
    yield attrgetter('layer4.0.conv2')
    yield attrgetter('layer4.0.downsample')
    yield attrgetter('layer4.1.conv1')
    yield attrgetter('layer4.1.conv2')


def ptq_resnet18(fresnet, in_bins: int, w_bins: int, symmetric_mode: bool, train_loader, device='cpu', mode='ada'):
    fresnet = fresnet.to(device)
    guide = MyQResNet([2, 2, 2, 2]).to(device)
    guide.load_state_dict(fresnet.state_dict())
    guide = guide.to(device)
    guide.eval()

    N_BLOCKS = 25
    cache = []
    for i, (X, _) in enumerate(train_loader):
        if i == N_BLOCKS:
            break
        cache.append(X.to(device))

    in_step = -1
    qresnet = MyQResNet([2, 2, 2, 2]).to(device)
    qresnet.load_state_dict(fresnet.state_dict())
    qresnet = qresnet.to(device)
    set_itter = resnet18_setter_iterator()
    for i, layer_getter in enumerate(resnet18_getter_iterator()):
        layer_setter = next(set_itter)
        qlayer = layer_getter(qresnet)
        guide_layer = layer_getter(guide)
        qlayer.cache_mode = True
        corrected_in = None
        guide_layer.cache_mode = True
        for X in cache:
            guide(X)
        if i > 0:
            for X in cache:
                qresnet(X)
            corrected_in = copy.deepcopy(qlayer.cached_in)
        qlayer.drop_cache()
        qlayer = QConv2d(guide_layer, in_bins, w_bins, symmetric_mode, in_step, mode, corrected_in).to(device)
        guide_layer.cache_mode = False
        guide_layer.drop_cache()
        if str(layer_getter)[-7:-2] == 'conv1':
            in_step = qlayer.input_quant.scale * qlayer.weight_quant.scale
        else:
            in_step = -1
        layer_setter(qresnet, qlayer)
    del guide
    return qresnet


def get_qresnet_18(original_model, pretrained=True, device='cpu'):
    model = MyQResNet([2, 2, 2, 2]).to(device)
    if pretrained:
        paired_layers = [
            (model.conv1.conv, original_model.conv1),
            (model.conv1.bn,   original_model.bn1),
            (model.layer1[0].conv1.conv, original_model.layer1[0].conv1),
            (model.layer1[0].conv1.bn,   original_model.layer1[0].bn1),
            (model.layer1[0].conv2.conv, original_model.layer1[0].conv2),
            (model.layer1[0].conv2.bn,   original_model.layer1[0].bn2),
            (model.layer1[1].conv1.conv, original_model.layer1[1].conv1),
            (model.layer1[1].conv1.bn,   original_model.layer1[1].bn1),
            (model.layer1[1].conv2.conv, original_model.layer1[1].conv2),
            (model.layer1[1].conv2.bn,   original_model.layer1[1].bn2),

            (model.layer2[0].conv1.conv, original_model.layer2[0].conv1),
            (model.layer2[0].conv1.bn,   original_model.layer2[0].bn1),
            (model.layer2[0].conv2.conv, original_model.layer2[0].conv2),
            (model.layer2[0].conv2.bn,   original_model.layer2[0].bn2),
            (model.layer2[0].downsample.conv, original_model.layer2[0].downsample[0]),
            (model.layer2[0].downsample.bn,   original_model.layer2[0].downsample[1]),
            (model.layer2[1].conv1.conv, original_model.layer2[1].conv1),
            (model.layer2[1].conv1.bn,   original_model.layer2[1].bn1),
            (model.layer2[1].conv2.conv, original_model.layer2[1].conv2),
            (model.layer2[1].conv2.bn,   original_model.layer2[1].bn2),

            (model.layer3[0].conv1.conv, original_model.layer3[0].conv1),
            (model.layer3[0].conv1.bn,   original_model.layer3[0].bn1),
            (model.layer3[0].conv2.conv, original_model.layer3[0].conv2),
            (model.layer3[0].conv2.bn,   original_model.layer3[0].bn2),
            (model.layer3[0].downsample.conv, original_model.layer3[0].downsample[0]),
            (model.layer3[0].downsample.bn,   original_model.layer3[0].downsample[1]),
            (model.layer3[1].conv1.conv, original_model.layer3[1].conv1),
            (model.layer3[1].conv1.bn,   original_model.layer3[1].bn1),
            (model.layer3[1].conv2.conv, original_model.layer3[1].conv2),
            (model.layer3[1].conv2.bn,   original_model.layer3[1].bn2),

            (model.layer4[0].conv1.conv, original_model.layer4[0].conv1),
            (model.layer4[0].conv1.bn,   original_model.layer4[0].bn1),
            (model.layer4[0].conv2.conv, original_model.layer4[0].conv2),
            (model.layer4[0].conv2.bn,   original_model.layer4[0].bn2),
            (model.layer4[0].downsample.conv, original_model.layer4[0].downsample[0]),
            (model.layer4[0].downsample.bn,   original_model.layer4[0].downsample[1]),
            (model.layer4[1].conv1.conv, original_model.layer4[1].conv1),
            (model.layer4[1].conv1.bn,   original_model.layer4[1].bn1),
            (model.layer4[1].conv2.conv, original_model.layer4[1].conv2),
            (model.layer4[1].conv2.bn,   original_model.layer4[1].bn2),

            (model.fc,   original_model.fc)
        ]

        for new_layer, old_layer in paired_layers:
            new_layer.load_state_dict(old_layer.state_dict())
        del original_model
    return model


def gradual_freeze_resnet18(qresnet, device):
    # FREEZE_ITERS = 100
    # TEST_ITERS = 50
    frozen_resnet = MyQResNet([2, 2, 2, 2], quantized=True).to(device)
    frozen_resnet.load_state_dict(qresnet.state_dict())
    frozen_resnet = frozen_resnet.to(device)

    train_stat = None
    test_stat = None
    set_itter = resnet18_setter_iterator(freeze=True)
    for layer_getter in resnet18_getter_iterator(freeze=True):
        layer_setter = next(set_itter)
        qlayer = layer_getter(qresnet)

        frozen_layer = FrozenConv2d(qlayer).to(device)
        layer_setter(frozen_resnet, frozen_layer)
        # loss = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(frozen_resnet.parameters(), lr=1e-5, momentum=0.9)
        #
        #
        # train_stat, test_stat = train(frozen_resnet, optimizer, train_loader, test_loader, loss,
        #                               n_epochs=1, train_stat=train_stat, test_stat=test_stat,
        #                               device=device, use_tqdm=True, print_results=True,
        #                               n_train_batches=FREEZE_ITERS, n_test_batches=TEST_ITERS
        #                               )
        # test(frozen_resnet, test_loader, loss, device, use_tqdm=True, n_batches=TEST_ITERS)
    return frozen_resnet, train_stat, test_stat


@click.command()
@click.option('-p', '--prefix', type=click.STRING, default='')
@click.option('--imagenet_train_path', type=click.STRING, default='./ILSVRC_kaggle/Data/CLS-LOC/train/')
@click.option('--imagenet_val_path', type=click.STRING, default='./ILSVRC_kaggle/Data/CLS-LOC/val_folders/')
@click.option('--pretrained_model', type=click.STRING, default=None)
# @click.option('--finetune', is_flag=True)
@click.option('--device', type=click.STRING, default='cuda')
@click.option('--n_exp_rep', type=click.INT, default=1)
def main(prefix, imagenet_train_path, imagenet_val_path, pretrained_model, device, n_exp_rep):
    BATCH_SIZE = 64

    train_ds = torchvision.datasets.ImageFolder(imagenet_train_path, transform=resnet_transformer_train)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_ds = torchvision.datasets.ImageFolder(imagenet_val_path, transform=resnet_transformer_test)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    device = torch.device(device)

    name = 'resnet18'
    if prefix:
        name = prefix + '_' + name
    if pretrained_model is not None:
        torch_resnet = resnet18()
        print(f'Loading {pretrained_model}')
        torch_resnet.load_state_dict(torch.load(pretrained_model))
    else:
        torch_resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    torch_resnet = torch_resnet.to(device)
    fqresnet = get_qresnet_18(torch_resnet, device=device)
    print('Loaded resnet')

    base_time = None
    # if finetune:
    #     print('starting finetuning')
    #     optimizer = optim.AdamW(fqresnet.parameters(), lr=5e-5, weight_decay=1e-5)
    #     loss = nn.CrossEntropyLoss()
    #     pretrain_start = time.time()
    #     train_stat, test_stat = train(fqresnet, optimizer, train_loader, test_loader, loss, n_epochs=1,
    #                                   device=device, use_tqdm=True, print_results=True)
    #     pretrain_end = time.time()
    #     save_statistics(f'{name}_baseline', train_stat, test_stat)
    #     base_time = pretrain_end - pretrain_start
    #     print('finetuning finished')

    save_model(fqresnet, f'{name}_baseline', test_loader, base_time, device=device)
    print('Starting quantization')
    for j in range(n_exp_rep):
        def quantization(in_bins, weight_bins, symmetric=True):
            print(f'Quantizing {in_bins}x{weight_bins}')
            quant_start = time.time()
            qqresnet = ptq_resnet18(fqresnet, in_bins, weight_bins, symmetric, train_loader, device=device)
            save_model(qqresnet, f'{name}_{in_bins}_{weight_bins}_ptq__{j + 1}', test_loader, device=device)
            frozen_resnet, train_stat, test_stat = gradual_freeze_resnet18(qqresnet, device=device)
            quant_end = time.time()
            save_model(frozen_resnet, f'{name}_{in_bins}_{weight_bins}_quant__{j + 1}', test_loader,
                       quant_end - quant_start, device)
            save_statistics(f'{name}_{in_bins}_{weight_bins}_quant__{j + 1}', train_stat, test_stat)

        quantization(29, 19, True)
        quantization(25, 21, True)
        quantization(23, 23, True)
        quantization(256, 256, False)
        quantization(16, 16, False)

    print('Done!')


if __name__ == '__main__':
    main()
