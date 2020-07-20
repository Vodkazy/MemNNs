#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/7/19 下午2:59
  @ Author   : Vodka
  @ File     : input_module .py
  @ Software : PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class InputModule(nn.Module):
    def __init__(self, word_embed, set_num, config):
        super(InputModule, self).__init__()
        self.config = config
        self.set_num = set_num
        self.word_embed = word_embed
        self.s_rnn_idim = config.word_embed_dim
        self.s_rnn = nn.GRU(self.s_rnn_idim, config.s_rnn_hdim, batch_first=True)

    def init_rnn_h(self, batch_size):
        return Variable(torch.zeros(
            self.config.s_rnn_ln * 1, batch_size, self.config.s_rnn_hdim)).to(self.config.device)

    def forward(self, stories, s_lens):
        """
        :param stories: batch_size * max_slen
        :param s_lens: batch_size * time_step(number of sentences of a story)
        :return: batch_size * time_step(number of sentences of a story) * hidden_size
        """
        word_tensors = F.dropout(self.word_embed(stories), self.config.word_dr)
        init_s_rnn_h = self.init_rnn_h(stories.size(0))
        gru_out, _ = self.s_rnn(word_tensors, init_s_rnn_h)
        gru_out = gru_out.contiguous().view(-1, self.config.s_rnn_hdim).cpu()
        # contiguous()这个函数，把tensor变成在内存中连续分布的形式
        # gru_out: (batch_size * time_step(number of sentences of a story)) * hidden_size
        s_lens_offset = (torch.arange(0, stories.size(0)).type(torch.LongTensor)
                         * self.config.max_slen[self.set_num]).unsqueeze(1)  # 获取每句话的起始位置,为了后面做加法的时候broadcast
        # s_lens_offset: batch_size * 1
        s_lens = (torch.clamp(s_lens + s_lens_offset - 1, min=0)).view(-1)  # 将输入input张量每个元素的夹紧到区间 [min,max]，并返回结果到一个新张量
        # s_lens: (batch_size*time_step)
        selected = gru_out[s_lens, :].view(-1, self.config.max_sentnum[self.set_num],
                                           self.config.s_rnn_hdim).to(self.config.device)
        # print("Input model forward...")
        return selected
