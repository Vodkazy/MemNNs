#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/7/19 下午2:59
  @ Author   : Vodka
  @ File     : question_module .py
  @ Software : PyCharm
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class QuestionModule(nn.Module):
    def __init__(self, word_embed, set_num, config):
        super(QuestionModule, self).__init__()
        self.config = config
        self.set_num = set_num
        self.word_embed = word_embed
        self.q_rnn_idim = config.word_embed_dim
        self.q_rnn = nn.GRU(self.q_rnn_idim, config.q_rnn_hdim, batch_first=True)

    def init_rnn_h(self, batch_size):
        return Variable(torch.zeros(self.config.s_rnn_ln * 1, batch_size, self.config.s_rnn_hdim)).to(
            self.config.device)

    def forward(self, questions, q_lens):
        """
        :param questions: batch_size * max_qlen
        :param q_lens: batch_size
        :return: batch_size * hidden_size
        """
        word_embed = F.dropout(self.word_embed(questions), self.config.word_dr)
        init_q_rnn_h = self.init_rnn_h(questions.size(0))
        gru_out, _ = self.q_rnn(word_embed, init_q_rnn_h)
        gru_out = gru_out.contiguous().view(-1, self.config.q_rnn_hdim).cpu()
        # gru_out.shape: (batch_size*timestep) * hidden_size
        q_lens = (torch.arange(0, questions.size(0)).type(torch.LongTensor)
                  * self.config.max_qlen[self.set_num] + q_lens - 1)
        # q_lens: batch_size
        selected = gru_out[q_lens, :].view(-1, self.config.q_rnn_hdim).to(self.config.device)
        # print("Question model forward...")
        return selected
