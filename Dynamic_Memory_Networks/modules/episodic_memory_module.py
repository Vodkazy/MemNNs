#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/7/19 下午2:59
  @ Author   : Vodka
  @ File     : episodic_memory_module .py
  @ Software : PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class EpisodicMemoryModule(nn.Module):
    def __init__(self, set_num, config):
        super(EpisodicMemoryModule, self).__init__()
        self.config = config
        self.set_num = set_num
        self.z_dim = config.s_rnn_hdim * 4
        self.e_cell_idim = config.s_rnn_hdim
        self.e_cell = nn.GRUCell(self.e_cell_idim, config.e_cell_hdim)
        self.g1 = nn.Linear(self.z_dim, config.g1_dim)
        self.g2 = nn.Linear(config.g1_dim, 1)

        # 原文z的计算方式：
        # self.z_sq = nn.Linear(config.s_rnn_hdim, config.q_rnn_hdim, bias=False)
        # self.z_sm = nn.Linear(config.s_rnn_hdim, config.m_cell_hdim, bias=False)
        # self.z_dim = config.s_rnn_hdim * 7 + 2

    def init_cell_h(self, batch_size):
        return Variable(torch.zeros(batch_size, self.config.s_rnn_hdim)).to(self.config.device)

    def forward(self, s_rep, q_rep, e_lens, memory):
        """
        :param s_rep: batch_size * time_step(number of sentences of a story) * hidden_size
        :param q_rep: batch_size * hidden_size
        :param e_lens: batch_size
        :param memory: batch_size * hidden_size
        :return: batch_size * hidden_size, batch_size * (time_step+1)
        """
        # expand s_rep to have sentinel
        sentinel = Variable(torch.zeros(s_rep.size(0), 1, self.config.s_rnn_hdim)).to(self.config.device)
        s_rep = torch.cat((s_rep, sentinel), 1)  # s_rep: batch_size*time_step*hidden_size; sentinel: _*1*_
        q_rep = q_rep.unsqueeze(1).expand_as(s_rep)  # 复制扩充
        memory = memory.unsqueeze(1).expand_as(s_rep)
        # s_rep, q_rep, memory: batch_size * time_step+1 * hidden_size
        """
        原文的z的计算方式
        sw = self.z_sq(s_rep.view(-1, self.config.s_rnn_hdim)).view(q_rep.size())
        swq = torch.sum(sw * q_rep, 2, keepdim=True)
        swm = torch.sum(sw * memory, 2, keepdim=True)
        Z = torch.cat([s_rep, memory, q_rep, s_rep*q_rep, s_rep*memory,
            torch.abs(s_rep-q_rep), torch.abs(s_rep-memory), swq, swm], 2)
        """
        Z = torch.cat([s_rep * q_rep, s_rep * memory, torch.abs(s_rep - q_rep), torch.abs(s_rep - memory)], 2).to(self.config.device)
        # Z: batch_size * time_step+1 * hidden_size*4
        G = self.g2(F.tanh(self.g1(Z.view(-1, self.z_dim))))
        # G: ((time_step+1)*batch_size) * 1
        G_s = F.sigmoid(G).view(-1, self.config.max_sentnum[self.set_num] + 1).unsqueeze(2)  # 在第2维度（下标为1）增加一个维度
        G_s = torch.transpose(G_s, 0, 1).contiguous()
        # G_s: (time_step+1) * batch_size * 1
        s_rep = torch.transpose(s_rep, 0, 1).contiguous()
        e_rnn_h = self.init_cell_h(s_rep.size(1)).to(self.config.device)
        # 对每个时间步的story进行前向传播，内部增强序列记忆
        hiddens = []
        for step, (gg, ss) in enumerate(zip(G_s, s_rep)):
            e_rnn_h = gg * self.e_cell(ss, e_rnn_h) + (1 - gg) * e_rnn_h
            # e_rnn_h: batch_size * hidden_size
            hiddens.append(e_rnn_h)
        hiddens = torch.transpose(torch.stack(hiddens), 0, 1).contiguous().view(-1, self.config.e_cell_hdim).cpu()
        # hiddens: ((time_step+1)*batch_size) * hidden_size
        e_lens = (torch.arange(0, s_rep.size(1)).type(torch.LongTensor)
                  * (self.config.max_sentnum[self.set_num] + 1) + e_lens - 1)
        # e_lens: batch_size
        selected = hiddens[e_lens, :].view(-1, self.config.e_cell_hdim).to(self.config.device)
        # print("Episodic Memory model forward...")
        return selected, G.view(-1, self.config.max_sentnum[self.set_num] + 1)
