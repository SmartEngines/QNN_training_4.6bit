import itertools
import torch
import torch.nn.functional as F
import numpy as np

from time import time
from tqdm import tqdm


@torch.no_grad()
def test_step(model, X, y:torch.Tensor, loss_function, device, k_max: int = 1):
    y = y.to(device)
    f = model(X.to(device))
    t_loss = loss_function(f, y).item()
    
    several_k = isinstance(k_max, list) or isinstance(k_max, tuple)
    if not several_k:
        k_maxes = [k_max]
    else:
        k_maxes = k_max
    
    corrects = []
    for i, ck in enumerate(k_maxes):
        if ck == 1:
            pred = torch.argmax(f, 1)
            correct = torch.sum(pred == y).item()
        elif ck > 1:
            pred = torch.topk(f, ck, 1).indices
            correct = torch.sum((y.view(-1, 1) == pred).int().sum(dim=1)).item()
        else:
            raise RuntimeError(f"incorrect number k_max: {k_max}")
        corrects.append(correct)
    if not several_k:
        corrects = corrects[0]
    return t_loss, corrects


@torch.no_grad()
def test(model, generator, loss_function=F.nll_loss, device=torch.device('cpu'),  k_max = 1, use_tqdm=False, n_batches: int = -1):
    several_k = isinstance(k_max, list) or isinstance(k_max, tuple)
    model.train(mode=False)
    correct = 0 if not several_k else np.zeros([len(k_max)])
    total = 0
    losses = []
    if n_batches > 0:
        generator = itertools.islice(generator, n_batches)
    if use_tqdm:
        for X, y in tqdm(generator, desc='testing'):
            b_loss, b_correct = test_step(model, X, y, loss_function, device, k_max)
            correct += b_correct
            total += len(y)
            losses.append(b_loss)
    else:
        for X, y in generator:
            b_loss, b_correct = test_step(model, X, y, loss_function, device, k_max)
            correct += b_correct
            total += len(y)
            losses.append(b_loss)
    accuracy = correct / total
    loss = np.mean(losses)
    return loss, accuracy


def train_step(model, opt, X, y, loss_function, device):
    opt.zero_grad()
    y = y.to(device)
    f = model(X.to(device))
    loss = loss_function(f, y)
    loss.backward()
    opt.step()

    pred = torch.argmax(f, 1)
    correct = torch.sum(pred == y).item()
    return loss.item(), correct


class Statistics:
    def __init__(self, generator_len, tb_writer, name=''):
        self.idx = []
        self.accuracy = []
        self.loss = []
        self.generator_len = generator_len
        self.tb_writer_ = tb_writer
        self.name = '/' + name if name else ''
        self.epoch = 0

    def append(self, loss, accuracy, global_step):
        self.idx.append(global_step / self.generator_len)
        self.accuracy.append(accuracy)
        self.loss.append(loss)

        if self.tb_writer_ is not None:
            self.tb_writer_.add_scalar(f'loss{self.name}', loss, global_step)
            self.tb_writer_.add_scalar(f'accuracy{self.name}', accuracy, global_step)


def train(model, opt, train_generator, test_generator, loss_function=F.nll_loss, n_epochs=10, scheduler=None,
          device=torch.device('cpu'), use_tqdm=False, tb_writer=None, print_results=False,
          epoch_callback=None, train_stat=None, test_stat=None, n_train_batches: int = -1, n_test_batches=-1):
    gen_len = len(train_generator)
    test_stat = Statistics(gen_len, tb_writer, 'test') if test_stat is None else test_stat
    train_stat = Statistics(gen_len, tb_writer, 'train') if train_stat is None else train_stat
    
    if n_train_batches > 0:
        train_generator = itertools.islice(train_generator, n_train_batches)

    global_step = train_stat.epoch * gen_len
    for epoch in range(n_epochs):
        start_time = time()
        model.train(True)
        generator = tqdm(train_generator, desc='training') if use_tqdm else train_generator
        for X, y in generator:
            b_loss, b_correct = train_step(model, opt, X, y, loss_function, device)
            b_acc = b_correct / len(y)
            global_step += 1

            train_stat.append(b_loss, b_acc, global_step)
        if scheduler is not None:
            scheduler.step()

        loss, acc = test(model, test_generator, loss_function=loss_function, use_tqdm=use_tqdm, device=device, n_batches=n_test_batches)
        test_stat.append(loss, acc, global_step)
        end_time = time()

        train_stat.epoch += 1
        test_stat.epoch += 1

        if print_results:
            print(f"epoch #{train_stat.epoch}: accuracy: {acc * 100 : .3f}%, loss: {loss: .5f}."
                  f" {end_time - start_time: .5f}s per epoch")

        if epoch_callback is not None:
            epoch_callback(train_stat, test_stat)

    return train_stat, test_stat
