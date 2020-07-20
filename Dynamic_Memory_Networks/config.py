#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/7/19 下午2:09
  @ Author   : Vodka
  @ File     : config .py
  @ Software : PyCharm
"""


class Config(object):
    def __init__(self):
        self.data_dir = './data/bAQbI'
        self.word2vec_type = 6  # 6 or 840 (B)
        self.word2vec_path = '/home/vodka/Glove/glove.' + str(self.word2vec_type) + 'B.300d.txt'
        self.word_embed_dim = 300
        self.batch_size = 32    # 预处理数据集时使用，以main中的设置为准
        self.max_sentnum = {}
        self.max_slen = {}
        self.max_qlen = {}
        self.max_episode = 5    # 预处理数据集时使用，以main中的设置为准
        self.word_vocab_size = 0
        self.save_preprocess = True
        self.preprocess_save_path = './data/babi(tmp).pkl'
        self.preprocess_load_path = './data/babi(10k).pkl'
