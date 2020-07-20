#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/7/18 下午4:46
  @ Author   : Vodka
  @ File     : model .py
  @ Software : PyCharm
"""
import os

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from modules.answer_module import AnswerModule
from modules.episodic_memory_module import EpisodicMemoryModule
from modules.input_module import InputModule
from modules.question_module import QuestionModule


class DMN(nn.Module):
    def __init__(self, config, idx2vec, set_num):
        super(DMN, self).__init__()
        self.config = config
        self.set_num = set_num

        # init word2vec
        self.word_embed = nn.Embedding(config.word_vocab_size, config.word_embed_dim, padding_idx=0)
        self.word_embed.weight.data.copy_(torch.from_numpy(np.array(idx2vec)))
        self.word_embed.weight.requires_grad = False

        self.input_module = InputModule(self.word_embed, self.set_num, self.config)
        self.question_module = QuestionModule(self.word_embed, self.set_num, self.config)
        self.episodic_memory_module = EpisodicMemoryModule(self.set_num, self.config)
        self.answer_module = AnswerModule(self.word_embed, self.set_num, self.config)
        self.m_cell = nn.GRUCell(config.e_cell_hdim, config.m_cell_hdim)
        self.m_cell.to(self.config.device)

        # define optim and loss
        params = self.model_params(debug=False)
        self.optimizer = optim.Adam(params, lr=config.lr)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, stories, questions, s_lens, q_lens, e_lens):
        s_rep = self.input_module(stories, s_lens)
        q_rep = self.question_module(questions, q_lens)
        memory = q_rep  # initial memory
        gates = []
        for episode in range(self.config.max_episode):
            e_rep, gate = self.episodic_memory_module(s_rep, q_rep, e_lens, memory)
            gates.append(gate)
            memory = self.m_cell(e_rep, memory)
        gates = torch.transpose(torch.stack(gates), 0, 1).contiguous()
        # 交换维度 变为 batch_size * max_episode * s_len+1
        outputs = self.answer_module(q_rep, memory)
        # print('memory', memory.size())
        # print('outputs', outputs.size())
        return outputs, gates

    def model_params(self, debug=True):
        print('model parameters: ', end='')
        params = []
        total_size = 0

        def multiply_iter(p_list):
            out = 1
            for p in p_list:
                out *= p
            return out

        for p in self.parameters():
            if p.requires_grad:
                params.append(p)
                total_size += multiply_iter(p.size())
            if debug:
                print(p.requires_grad, p.size())
        print('%s\n' % '{:,}'.format(total_size))
        return params

    def get_regloss(self, weight_decay=None):
        if weight_decay is None:
            weight_decay = self.config.wd
        reg_loss = 0
        params = []  # add params here
        for param in params:
            reg_loss += torch.norm(param.weight, 2)
        return reg_loss * weight_decay

    def decay_lr(self, lr_decay=None):
        if lr_decay is None:
            lr_decay = self.config.lr_decay
        self.config.lr /= lr_decay
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config.lr
        print('\tlearning rate decay to %.3f' % self.config.lr)

    def get_metrics(self, outputs, targets, multiple=False):
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
                max_idx = torch.max(outputs, 1)[1].data.numpy()
                outputs_topk = torch.topk(outputs, 3)[1].data.numpy()
                targets = targets.data.numpy()

                topk_list.append(outputs_topk)
                target_list.append(targets)

            acc = np.array([1.0 for _ in range(outputs.size(0))])
            for target, topk in zip(target_list, topk_list):
                acc *= np.array([float(k == tk[0] or k == -100) \
                                 for (k, tk) in zip(target, topk)])
                # print(acc)
            acc = np.mean(acc) * 100

        return acc

    def save_checkpoint(self, state, filename=None):
        if filename is None:
            filename = (self.config.checkpoint_dir + \
                        self.config.model_name + str(self.set_num) + '.pth')
        else:
            filename = self.config.checkpoint_dir + filename
        print('\t=> save checkpoint %s' % filename)
        torch.save(state, filename)

    def load_checkpoint(self, filename=None):
        if filename is None:
            filename = (self.config.checkpoint_dir + \
                        self.config.model_name + str(self.set_num) + '.pth')
        else:
            filename = self.config.checkpoint_dir + filename
        print('\t=> load checkpoint %s' % filename)
        checkpoint = torch.load(filename)
        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        # self.config = checkpoint['config']

    # def load_model(self, load_model_path, device):
    #     if os.path.exists(load_model_path):
    #         self.load_state_dict(torch.load(load_model_path, map_location=device))
    #         self.to(device)
    #         print("Load over !")
    #     else:
    #         print("No such file !")
    #
    # def save_model(self, save_model_path):
    #     torch.save(self.state_dict(), save_model_path)