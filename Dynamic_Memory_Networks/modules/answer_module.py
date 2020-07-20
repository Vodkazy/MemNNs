#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/7/19 下午2:59
  @ Author   : Vodka
  @ File     : answer_module .py
  @ Software : PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnswerModule(nn.Module):
    def __init__(self, word_embed, set_num, config):
        super(AnswerModule, self).__init__()
        self.config = config
        self.set_num = set_num
        self.word_embed = word_embed
        self.a_cell_idim = config.q_rnn_hdim + config.word_vocab_size
        self.a_cell = nn.GRUCell(self.a_cell_idim, config.a_cell_hdim)
        self.out = nn.Linear(config.a_cell_hdim, config.word_vocab_size, bias=False)

    def forward(self, q_rep, memory):
        """
        :param q_rep: batch_size * hidden_size
        :param memory: batch_size * hidden_size
        :return: batch_size * max_alen * word_vocab_size
        """
        y = F.softmax(self.out(memory))
        a_rnn_h = memory
        ys = []
        for step in range(self.config.max_alen):
            a_rnn_h = self.a_cell(torch.cat((y, q_rep), 1), a_rnn_h)
            # a_rnn_h: batch_size * hidden_size
            z = self.out(a_rnn_h)
            # z: batch_size * word_vocab_size
            y = F.softmax(z)
            ys.append(z)
        ys = torch.transpose(torch.stack(ys), 0, 1).contiguous()
        """
        z = self.out(torch.cat((memory, q_rep), 1))
        ys = torch.transpose(torch.stack([z]), 0, 1).contiguous()
        """
        # print("Answer model forward...")
        return ys
