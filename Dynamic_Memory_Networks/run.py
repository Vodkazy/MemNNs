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
from torch.autograd import Variable


def progress(_progress):
    bar_length = 5  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(_progress, int):
        _progress = float(_progress)
    if not isinstance(_progress, float):
        _progress = 0
        status = "error: progress var must be float\r\n"
    if _progress < 0:
        _progress = 0
        status = "Halt...\r\n"
    if _progress >= 1:
        _progress = 1
        status = ""
    block = int(round(bar_length * _progress))
    text = "\r\t[%s]\t%.2f%% %s" % (
        "#" * block + " " * (bar_length - block), _progress * 100, status)

    return text


def run_experiment(model, dataset, set_num, device):
    best_metric = np.zeros(2)
    early_stop = False
    if model.config.train:
        if model.config.resume:
            model.load_checkpoint()

        for ep in range(model.config.epoch):
            if early_stop:
                break
            print('- Training Epoch %d' % (ep + 1))
            run_epoch(model, dataset, ep, 'tr', set_num, True, device)

            if model.config.valid:
                print('- Validation')
                met = run_epoch(model, dataset, ep, 'va', set_num, False, device)
                if best_metric[1] < met[1]:
                    best_metric = met
                    model.save_checkpoint({
                        'config': model.config,
                        'state_dict': model.state_dict(),
                        'optimizer': model.optimizer.state_dict()
                    })
                    if best_metric[1] == 100:
                        break
                else:
                    # model.decay_lr()
                    if model.config.early_stop:
                        early_stop = True
                        print('\tearly stop applied')
                print('\tbest metrics:\t%s' % ('\t'.join(['{:.2f}'.format(k)
                                                          for k in best_metric])))

            if model.config.test:
                print('- Testing')
                run_epoch(model, dataset, ep, 'te', set_num, False, device)
            print()

    if model.config.test:
        print('- Load Validation/Testing')
        if model.config.resume or model.config.train:
            model.load_checkpoint()
        run_epoch(model, dataset, 0, 'va', set_num, False, device)
        run_epoch(model, dataset, 0, 'te', set_num, False, device)
        print()

    return best_metric


def run_epoch(m, d, ep, mode='tr', set_num=1, is_train=True, device=torch.device('cpu')):
    total_metrics = np.zeros(2)
    total_step = 0.0
    print_step = m.config.print_step
    start_time = datetime.now()
    d.shuffle_data(seed=None, mode='tr')
    while True:
        m.optimizer.zero_grad()
        stories, questions, answers, sup_facts, s_lens, q_lens, e_lens = \
            d.get_next_batch(mode, set_num)
        # d.decode_data(stories[0], questions[0], answers[0], sup_facts[0], s_lens[0])
        wrap_tensor = lambda x: torch.LongTensor(np.array(x))
        wrap_var = lambda x: Variable(wrap_tensor(x)).to(device)
        stories = wrap_var(stories)
        questions = wrap_var(questions)
        answers = wrap_var(answers)
        sup_facts = wrap_var(sup_facts) - 1  # 原来的super是1base，现在改为0base
        s_lens = wrap_tensor(s_lens)
        q_lens = wrap_tensor(q_lens)
        e_lens = wrap_tensor(e_lens)

        if is_train:
            m.train()
        else:
            m.eval()
        outputs, gates = m(stories, questions, s_lens, q_lens, e_lens)
        a_loss = m.criterion(outputs[:, 0, :], answers[:, 0])
        if answers.size(1) > 1:  # multiple answer
            for ans_idx in range(m.config.max_alen):
                a_loss += m.criterion(outputs[:, ans_idx, :], answers[:, ans_idx])
        for episode in range(5):
            if episode == 0:
                g_loss = m.criterion(gates[:, episode, :], sup_facts[:, episode])
            else:
                g_loss += m.criterion(gates[:, episode, :], sup_facts[:, episode])
        beta = 0 if ep < m.config.beta_cnt and mode == 'tr' else 1
        alpha = 1
        metrics = m.get_metrics(outputs, answers, multiple=answers.size(1) > 1)
        total_loss = alpha * g_loss + beta * a_loss

        if is_train:
            total_loss.backward()
            nn.utils.clip_grad_norm(m.parameters(), m.config.grad_max_norm)  # 梯度裁剪
            m.optimizer.step()

        total_metrics[0] += total_loss.item()
        total_metrics[1] += metrics
        total_step += 1.0

        # print step
        if d.get_batch_ptr(mode) % print_step == 0 or total_step == 1:
            et = int((datetime.now() - start_time).total_seconds())
            _progress = progress(
                d.get_batch_ptr(mode) / d.get_dataset_len(mode, set_num))
            if d.get_batch_ptr(mode) == 0:
                _progress = progress(1)
            _progress += '[%s] time: %s' % (
                '\t'.join(['{:.2f}'.format(k)
                           for k in total_metrics / total_step]),
                '{:2d}:{:2d}:{:2d}'.format(et // 3600, et % 3600 // 60, et % 60))
            sys.stdout.write(_progress)
            sys.stdout.flush()

            # end of an epoch
            if d.get_batch_ptr(mode) == 0:
                et = (datetime.now() - start_time).total_seconds()
                print('\n\ttotal metrics:\t%s' % ('\t'.join(['{:.2f}'.format(k)
                                                             for k in total_metrics / total_step])))
                break

    return total_metrics / total_step
