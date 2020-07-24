#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/7/19 下午2:09
  @ Author   : Vodka
  @ File     : config .py
  @ Software : PyCharm
"""
from os.path import expanduser


class Config(object):
    def __init__(self):
        """
        数据预处理参数列表：
            data_dir                数据集路径
            word2vec_type           6B tokens or 840 (B)
            word2vec_path           预训练glove路径
            word_embed_dim          词向量维度
            batch_size              批处理大小
            max_sentnum             每个训练集的的最多句子个数的词典
            max_slen                每个训练集的的最长fact长度的词典
            max_qlen                每个训练集的的最长问句长度的词典
            max_episode             记忆模块最大迭代次数
            word_vocab_size         新词向量词典大小
            save_preprocess         是否保存处理
            preprocess_save_path    预处理保存路径
            preprocess_load_path    预处理加载路径
        """
        self.data_dir = './data/bAQbI'
        self.word2vec_type = 6  # 6 or 840 (B)
        self.word2vec_path = expanduser('~') + '/Glove/glove.' + str(self.word2vec_type) + 'B.300d.txt'
        self.word_embed_dim = 300
        self.batch_size = 32  # 预处理数据集时使用，以main中的设置为准
        self.max_sentnum = {}
        self.max_slen = {}
        self.max_qlen = {}
        self.max_episode = 10  # 预处理数据集时使用，以main中的设置为准
        self.word_vocab_size = 0
        self.load_preprocess = False
        self.preprocess_path = './data/babi.pkl'
