#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/7/19 下午3:50
  @ Author   : Vodka
  @ File     : run .py
  @ Software : PyCharm
"""
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn


def train(model, dataset, epoch, set_num, device):
    """
    训练模型
    :param model:       模型
    :param dataset:     数据集
    :param epoch:       轮数序号
    :param set_num:     集合序号
    :param device:      设备，cpu/gpu
    :return:
    """
    print('- Training   Epoch %d' % (epoch + 1))
    run(model, dataset, epoch, 'tr', set_num, device)


def valid(model, dataset, epoch, set_num, device, best_metric):
    """
    验证模型
    :param model:       模型
    :param dataset:     数据集
    :param epoch:       轮数序号
    :param set_num:     集合序号
    :param device:      设备，cpu/gpu
    :param best_metric: 当前最佳指标
    :return:            本轮过后的最佳指标
    """
    early_stop = False
    if epoch == -1:
        metric = run(model, dataset, 0, 'va', set_num, device)
    else:
        print('- Validation Epoch %d' % (epoch + 1))
        metric = run(model, dataset, epoch, 'va', set_num, device)
    if best_metric[1] < metric[1]:
        best_metric = metric
        model.save_checkpoint({
            'config': model.config,
            'state_dict': model.state_dict(),
            'optimizer': model.optimizer.state_dict()
        })
    else:
        if model.config.lr_decay != 1.0:
            model.decay_lr()
        if model.config.early_stop and best_metric[1] == 100:
            early_stop = True
            print('\tearly stop applied')
    print('\tbest metrics:\t%s' % ('\t'.join(['{:.2f}'.format(k)
                                              for k in best_metric])))
    return best_metric, early_stop


def test(model, dataset, epoch, set_num, device):
    """
    测试模型
    :param model:       模型
    :param dataset:     数据集
    :param epoch:       轮数序号
    :param set_num:     集合序号
    :param device:      设备，cpu/gpu
    :return:
    """
    if epoch == -1:
        run(model, dataset, 0, 'te', set_num, device)
    else:
        print('- Testing    Epoch %d' % (epoch + 1))
        run(model, dataset, epoch, 'te', set_num, device)


def get_metrics(outputs, targets, multiple=False):
    """
    计算评价指标
    :param outputs:     模型输出
    :param targets:     目标值
    :param multiple:    是否为多答案
    :return:            评价指标，accuracy
    """
    if not multiple:
        outputs = outputs[:, 0, :]
        targets = targets[:, 0]
        max_idx = torch.max(outputs, 1)[1].data.cpu().numpy()
        outputs_topk = torch.topk(outputs, 3)[1].data.cpu().numpy()
        targets = targets.data.cpu().numpy()

        acc = np.mean([float(k == tk[0]) for (k, tk)
                       in zip(targets, outputs_topk)]) * 100
    else:
        topk_list = []
        target_list = []
        o_outputs = outputs[:]
        o_targets = targets[:]
        for idx in range(outputs.size(1)):
            outputs = o_outputs[:, idx, :]
            targets = o_targets[:, idx]
            max_idx = torch.max(outputs, 1)[1].data.cpu().numpy()
            outputs_topk = torch.topk(outputs, 3)[1].cpu().data.numpy()
            targets = targets.data.cpu().numpy()
            topk_list.append(outputs_topk)
            target_list.append(targets)

        acc = np.array([1.0 for _ in range(outputs.size(0))])
        for target, topk in zip(target_list, topk_list):
            acc *= np.array([float(k == tk[0] or k == -100) \
                             for (k, tk) in zip(target, topk)])
        acc = np.mean(acc) * 100

    return acc


def run(model, data, epoch, mode='tr', set_num=1, device=torch.device('cpu')):
    """
    模型运行
    :param model:       模型
    :param data:        数据集
    :param epoch:       轮数序号
    :param mode:        模式，tr:训练；va:验证；te:测试
    :param set_num:     集合序号
    :param device:      设备，cpu/gpu
    :return:            实时loss
    """
    total_metrics = np.zeros(2)
    total_step = 0.0
    """
    print_step = model.config.print_step
    start_time = datetime.now()
    data.shuffle_data(seed=None, mode=mode)
    # 旧式写法，while循环配合data.get_batch使用，一次只返回一个batch
    while True:
        stories, questions, answers, sup_facts, s_lens, q_lens, e_lens = \
            data.get_batch(mode, set_num)
        model.optimizer.zero_grad()
        wrap_tensor = lambda x: torch.LongTensor(np.array(x))
        from torch.autograd import Variable
        wrap_var = lambda x: Variable(wrap_tensor(x)).to(device)
        stories = wrap_var(stories)
        questions = wrap_var(questions)
        answers = wrap_var(answers)
        sup_facts = wrap_var(sup_facts) - 1  # 原来的super是1base，现在改为0base
        s_lens = wrap_tensor(s_lens)
        q_lens = wrap_tensor(q_lens)
        e_lens = wrap_tensor(e_lens)

        if mode == 'tr':
            model.train()
        else:
            model.eval()

        outputs, gates = model(stories, questions, s_lens, q_lens, e_lens)
        a_loss = model.criterion(outputs[:, 0, :], answers[:, 0])
        if answers.size(1) > 1:  # multiple answer
            for ans_idx in range(model.config.max_alen):
                a_loss += model.criterion(outputs[:, ans_idx, :], answers[:, ans_idx])
        for episode in range(5):
            if episode == 0:
                g_loss = model.criterion(gates[:, episode, :], sup_facts[:, episode])
            else:
                g_loss += model.criterion(gates[:, episode, :], sup_facts[:, episode])
        beta = 0 if epoch < model.config.beta_cnt and mode == 'tr' else 1
        alpha = 1
        metrics = get_metrics(outputs, answers, multiple=answers.size(1) > 1)
        total_loss = alpha * g_loss + beta * a_loss

        if mode == 'tr':
            total_loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), model.config.grad_max_norm)  # 梯度裁剪
            model.optimizer.step()

        total_metrics[0] += total_loss.item()
        total_metrics[1] += metrics
        total_step += 1.0

        # print step
        if data.get_batch_id(mode) % print_step == 0 or total_step == 1:
            et = int((datetime.now() - start_time).total_seconds())

            def progress(x):
                bar_length = 5  # Modify this to change the length of the progress bar
                status = ""
                if isinstance(x, int):
                    x = float(x)
                if not isinstance(x, float):
                    x = 0
                    status = "error: progress var must be float\r\n"
                if x < 0:
                    x = 0
                    status = "Halt...\r\n"
                if x >= 1:
                    x = 1
                    status = ""
                block = int(round(bar_length * x))
                text = "\r\t[%s]\t%.2f%% %s" % (
                    "#" * block + " " * (bar_length - block), x * 100, status)
                return text

            _progress = progress(data.get_batch_id(mode) / data.get_dataset_len(mode, set_num))
            if data.get_batch_id(mode) == 0:
                _progress = progress(1)
            _progress += '[%s] time: %s' % ('\t'.join(['{:.2f}'.format(k) for k in total_metrics / total_step]),
                                            '{:2d}:{:2d}:{:2d}'.format(et // 3600, et % 3600 // 60, et % 60))
            sys.stdout.write(_progress)
            sys.stdout.flush()

            # end of an epoch
            if data.get_batch_id(mode) == 0:
                print('\n\ttotal metrics:\t%s' % ('\t'.join(['{:.2f}'.format(k)
                                                             for k in total_metrics / total_step])))
                break
    # """
    # """
    # 迭代式写法，复写Dataloader，迭代式读取
    dataloader = data.get_batch_dataloader(mode, set_num)
    _len = len(dataloader)
    from tqdm import tqdm
    for _index, (stories, questions, answers, sup_facts, s_lens, q_lens, e_lens) in enumerate(tqdm(dataloader)):
        model.optimizer.zero_grad()
        stories, questions, answers, sup_facts, s_lens, q_lens, e_lens = \
            torch.LongTensor(stories).to(device), torch.LongTensor(questions).to(device), \
            torch.LongTensor(answers).to(device), torch.LongTensor(sup_facts).to(device), \
            torch.LongTensor(s_lens), torch.LongTensor(q_lens), torch.LongTensor(e_lens)
        if mode == 'tr':
            model.train()
        else:
            model.eval()

        outputs, gates = model(stories, questions, s_lens, q_lens, e_lens)
        a_loss = model.criterion(outputs[:, 0, :], answers[:, 0])
        if answers.size(1) > 1:  # multiple answer
            for ans_idx in range(model.config.max_alen):
                a_loss += model.criterion(outputs[:, ans_idx, :], answers[:, ans_idx])
        for episode in range(5):
            if episode == 0:
                g_loss = model.criterion(gates[:, episode, :], sup_facts[:, episode])
            else:
                g_loss += model.criterion(gates[:, episode, :], sup_facts[:, episode])
        beta = 0 if epoch < model.config.beta_cnt and mode == 'tr' else 1
        alpha = 1
        metrics = get_metrics(outputs, answers, multiple=answers.size(1) > 1)
        total_loss = alpha * g_loss + beta * a_loss

        if mode == 'tr':
            total_loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), model.config.grad_max_norm)  # 梯度裁剪
            model.optimizer.step()

        total_metrics[0] += total_loss.item()
        total_metrics[1] += metrics
        total_step += 1.0

    print('\ttotal metrics:\t%s' % ('\t'.join(['{:.2f}'.format(k)
                                                 for k in total_metrics / total_step])))
    # """
    return total_metrics / total_step


def experiment(model, dataset, set_num, device):
    """
    进行实验的入口，训练、验证、测试集成
    :param model:       模型
    :param dataset:     数据集
    :param set_num:     集合序号
    :param device:      设备，cpu/gpu
    :return:            评价指标，(loss，accuracy）
    """
    best_metric = np.zeros(2)
    early_stop = False
    if model.config.train:
        if model.config.resume:
            model.load_checkpoint()

        for epoch in range(model.config.epoch):
            if early_stop:
                break
            train(model, dataset, epoch, set_num, device)

            if model.config.valid:
                best_metric, early_stop = valid(model, dataset, epoch, set_num, device, best_metric)

            if model.config.test:
                test(model, dataset, epoch, set_num, device)
            print()

    if model.config.test:
        print('===== Load Validation/Testing =====')
        if model.config.resume or model.config.train:
            model.load_checkpoint()
        valid(model, dataset, -1, set_num, device, best_metric)
        test(model, dataset, -1, set_num, device)
        print()

    return best_metric
