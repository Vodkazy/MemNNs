#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/7/18 下午5:18
  @ Author   : Vodka
  @ File     : dataset .py
  @ Software : PyCharm
"""
import copy
import os
import pickle

import nltk
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as DT

from config import Config


class MyDataset(DT):
    """
    复写DataSet，返回dataloader迭代器
    """

    def __init__(self, dataset, set_num, config, word2idx):
        super(MyDataset, self).__init__()
        self.dataset = dataset
        self.set_num = set_num
        self.config = config
        self.word2idx = word2idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_item = self.dataset[index]
        story = data_item[0]
        question = data_item[1]
        answer = data_item[2]
        if len(answer) < 1:
            while len(answer) != self.config.max_alen:
                answer.append(-100)
        sup_fact = data_item[3]
        while len(sup_fact) < self.config.max_episode:
            sup_fact.append(self.config.max_sentnum[self.set_num] + 1)
        s_len = [idx + 1 for idx, val in enumerate(story) if val == self.word2idx['.']]
        e_len = len(s_len)
        while len(s_len) != self.config.max_sentnum[self.set_num]:
            s_len.append(0)
        q_len = [idx + 1 for idx, val in enumerate(question) if val == self.word2idx['?']][0]
        return (story, question, answer, sup_fact,
                s_len, q_len, e_len)


class Dataset(object):
    """
    自己手撸数据集，封装自制函数
    """

    def __init__(self, config):
        self.config = config
        # init_settings
        self.dataset = {}
        self.train_id = 0
        self.valid_id = 0
        self.test_id = 0
        # init_dict
        self.PAD = 'PAD'
        self.word2idx = {}
        self.idx2word = {}
        self.idx2vec = []  # pretrained
        self.word2idx[self.PAD] = 0
        self.idx2word[0] = self.PAD
        self.init_word_dict = {}  # 遍历所有数据集，构造初始词典 word : (word_idx, word_cnt)

    def preprocess(self):
        """
        预处理数据集
        :return:
        """
        print("====== Preprocessing ======")
        if not os.path.exists('./data'):
            os.makedirs('./data')
        if not os.path.exists('./results'):
            os.makedirs('./results')
        self.build_corpus_dict(self.config.data_dir)
        self.get_glove_and_update_word2vec(self.config.word2vec_path)
        self.process_corpus(self.config.data_dir)
        pickle.dump(self, open(self.config.preprocess_path, 'wb'))
        print("====== Preprocessing over ======")

    ######################
    #    构建数据集操作    #
    ######################
    def update_word_dict(self, key):
        """
        更新word2vec字典
        :param key:         单词
        :return:
        """
        if key not in self.word2idx:
            self.word2idx[key] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = key

    def map_word2idx(self, key_list, dictionary):
        """
        将单词序列映射为id序列
        :param key_list:    单词序列
        :param dictionary:  word2vec字典
        :return:            id序列
        """
        output = []
        for key in key_list:
            assert key in dictionary
            if key in dictionary:
                output.append(dictionary[key])
        return output

    def build_corpus_dict(self, dir):
        """
        根据数据集语料创建单词词典
        :param dir:         数据集路径
        :return:
        """
        print('### building word dict %s' % dir)
        for subdir, _, files, in os.walk(dir):
            for file in sorted(files):
                with open(os.path.join(subdir, file)) as f:
                    for line_idx, line in enumerate(f):
                        line = line[:-1]
                        story_idx = int(line.split(' ')[0])

                        def update_init_dict(split):
                            for word in split:
                                if word not in self.init_word_dict:
                                    self.init_word_dict[word] = (
                                        len(self.init_word_dict), 1)  # 第一维idx，第二维次数
                                else:
                                    self.init_word_dict[word] = (
                                        self.init_word_dict[word][0],
                                        self.init_word_dict[word][1] + 1)

                        if '\t' in line:  # question
                            question, answer, _ = line.split('\t')
                            question = ' '.join(question.split(' ')[1:])
                            q_split = nltk.word_tokenize(question)
                            if self.config.word2vec_type == 6:
                                q_split = [w.lower() for w in q_split]
                            update_init_dict(q_split)

                            answer = answer.split(',') if ',' in answer else [answer]
                            if self.config.word2vec_type == 6:
                                answer = [w.lower() for w in answer]
                            update_init_dict(answer)
                            # TODO: check vocab
                            """
                            for a in answer: 
                                if a not in self.init_word_dict:
                                    print(a)
                            """
                        else:  # story
                            story_line = ' '.join(line.split(' ')[1:])
                            s_split = nltk.word_tokenize(story_line)
                            if self.config.word2vec_type == 6:
                                s_split = [w.lower() for w in s_split]
                            update_init_dict(s_split)

        print('init dict size', len(self.init_word_dict))
        # print(self.init_word_dict)

    def get_glove_and_update_word2vec(self, path):
        """
        获得预训练glove词向量，并以此更新当前自制的word2vec
        :param path:        glove文件路径
        :return:
        """
        print('\n### loading pretrained %s' % path)
        word2vec = {}
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            while True:
                try:
                    line = f.readline()
                    if not line:
                        break
                    word = line.split()[0]
                    if word == '40000':
                        continue
                    vec = [float(l) for l in line.split()[1:]]
                    word2vec[word] = vec
                except ValueError as e:
                    print(e)

        unk_cnt = 0
        self.idx2vec.append([0.0] * self.config.word_embed_dim)  # PAD

        for word, (word_idx, word_cnt) in self.init_word_dict.items():
            if word != 'UNK' and word != 'PAD':
                assert word_cnt > 0
                if word in word2vec:
                    self.update_word_dict(word)  # 用初始词典里击中glove的单词创造新词典
                    self.idx2vec.append(word2vec[word])
                else:
                    unk_cnt += 1
        # print('apple:', self.word2idx['apple'], word2vec['apple'][:5])
        # print('apple:', self.idx2vec[self.word2idx['apple']][:5])
        print('pretrained vectors', np.asarray(self.idx2vec).shape, 'unk', unk_cnt)
        print('dictionary change', len(self.init_word_dict),
              'to', len(self.word2idx), len(self.idx2word))

    def process_corpus(self, dir):
        """
        预处理数据集语料，将train/valid/test分类
        :param dir:         数据集路径
        :return:
        """
        dataset_statistics = []
        print('\n### processing %s' % dir)
        for subdir, _, files, in os.walk(dir):
            for file in sorted(files):
                with open(os.path.join(subdir, file)) as f:
                    max_sentnum = 0
                    max_slen = 0
                    max_qlen = 0
                    qa_num = file.split('_')[0][2:]
                    set_type = file.split('_')[-1][:-4]
                    story_list = []
                    sf_cnt = 1
                    si2sf = {}
                    total_data = []

                    for line_idx, line in enumerate(f):
                        line = line[:-1]
                        story_idx = int(line.split(' ')[0])
                        if story_idx == 1:  # index为1的代表新故事开始
                            story_list = []
                            sf_cnt = 1
                            si2sf = {}

                        if '\t' in line:  # question
                            question, answer, sup_fact = line.split('\t')
                            question = ' '.join(question.split(' ')[1:])
                            q_split = nltk.word_tokenize(question)
                            if self.config.word2vec_type == 6:
                                q_split = [w.lower() for w in q_split]
                            q_split = self.map_word2idx(q_split, self.word2idx)

                            answer = answer.split(',') if ',' in answer else [answer]
                            if self.config.word2vec_type == 6:
                                answer = [w.lower() for w in answer]
                            answer = self.map_word2idx(answer, self.word2idx)
                            sup_fact = [si2sf[int(sf)] for sf in sup_fact.split()]

                            sentnum = story_list.count(self.word2idx['.'])
                            max_sentnum = max_sentnum if max_sentnum > sentnum \
                                else sentnum
                            max_slen = max_slen if max_slen > len(story_list) \
                                else len(story_list)
                            max_qlen = max_qlen if max_qlen > len(q_split) \
                                else len(q_split)

                            story_tmp = story_list[:]
                            total_data.append([story_tmp, q_split, answer, sup_fact])

                        else:  # story
                            story_line = ' '.join(line.split(' ')[1:])
                            s_split = nltk.word_tokenize(story_line)
                            if self.config.word2vec_type == 6:
                                s_split = [w.lower() for w in s_split]
                            s_split = self.map_word2idx(s_split, self.word2idx)
                            story_list += s_split
                            si2sf[story_idx] = sf_cnt
                            sf_cnt += 1

                    self.dataset[str(qa_num) + '_' + set_type] = total_data

                    def check_update(d, k, v):
                        if k in d:
                            d[k] = v if v > d[k] else d[k]
                        else:
                            d[k] = v

                    check_update(self.config.max_sentnum, int(qa_num), max_sentnum)
                    check_update(self.config.max_slen, int(qa_num), max_slen)
                    check_update(self.config.max_qlen, int(qa_num), max_qlen)
                    self.config.word_vocab_size = len(self.word2idx)

                    dataset_statistics.append({'qa_num': str(qa_num),
                                               'data size': len(total_data),
                                               'max sentnum': max_sentnum,
                                               'max slen': max_slen,
                                               'max qlen': max_qlen
                                               })
        print(dataset_statistics)

    def pad_sent_word(self, sentword, maxlen):
        """
        将不足maxlen的单个句子id序列padding成等长
        :param sentword:    句子序列
        :param maxlen:      单个句子最大长度
        :return:            padding之后的句子
        """
        while len(sentword) != maxlen:
            sentword.append(self.word2idx[self.PAD])
        return sentword

    def pad_dataset(self, dataset, set_num):
        """
        把数据集中所有数据padding成等长
        :param dataset:     数据集
        :param set_num:     数据集序号
        :return:            padding之后的数据集
        """
        for data in dataset:
            # 要把所有的句子padding成一样长的大小
            story, question, _, _ = data
            self.pad_sent_word(story, self.config.max_slen[set_num])
            self.pad_sent_word(question, self.config.max_qlen[set_num])

        return dataset

    def valid_data(self, mode='tr'):
        """
        验证数据集是否生成正确
        :param mode:        数据集模式，train/valid/test
        :return:
        """
        for set_num in range(1):
            if mode == 'tr':
                self.shuffle_data(mode, set_num + 1)
                while True:
                    s, q, a, sf, sl, ql, el = self.get_batch(mode, set_num + 1, batch_size=1000)
                    print(self.get_batch_id(mode), len(s))
                    if self.get_batch_id(mode) == 0:
                        self.decode_data(s[0], q[0], a[0], sf[0], sl[0][:el[0]])
                        print('iteration test pass!', mode)
                        break
            if mode == 'va':
                self.shuffle_data(mode, set_num + 1)
                while True:
                    s, q, a, sf, sl, ql, el = self.get_batch(mode, set_num + 1, batch_size=100)
                    print(self.get_batch_id(mode), len(s))
                    if self.get_batch_id(mode) == 0:
                        self.decode_data(s[0], q[0], a[0], sf[0], sl[0][:el[0]])
                        print('iteration test pass!', mode)
                        break
            if mode == 'te':
                self.shuffle_data(mode, set_num + 1)
                while True:
                    s, q, a, sf, sl, ql, el = self.get_batch(
                        mode, set_num + 1, batch_size=100)
                    if self.get_batch_id(mode) == 0:
                        self.decode_data(s[0], q[0], a[0], sf[0], sl[0][:el[0]])
                        print('iteration test pass!', mode)
                        break

    ######################
    #     获取数据操作     #
    ######################

    def init_batch_id(self, mode=None):
        """
        初始化指向batch序号的模拟指针
        :param mode:        数据集模式，train/valid/test
        :return:
        """
        if mode is None:
            self.train_id = 0
            self.valid_id = 0
            self.test_id = 0
        elif mode == 'tr':
            self.train_id = 0
        elif mode == 'va':
            self.valid_id = 0
        elif mode == 'te':
            self.test_id = 0

    def get_batch_id(self, mode):
        """
        获取当前指向的batch的序号
        :param mode:        数据集模式，train/valid/test
        :return:            当前batch序号
        """
        if mode == 'tr':
            return self.train_id
        elif mode == 'va':
            return self.valid_id
        elif mode == 'te':
            return self.test_id

    def get_batch(self, mode='tr', set_num=1, batch_size=None):
        """
        对数据进行处理，获取一个batch的数据
        此处数据预处理比较复杂，因此采用手写get_batch的方式，简单的可以直接通过dataloader返回一个迭代对象
        :param mode:        数据集模式，train/valid/test
        :param set_num:     数据集序号
        :param batch_size:  批处理大小
        :return:            一个batch的数据
        """
        if batch_size is None:
            batch_size = self.config.batch_size

        if mode == 'tr':
            id = self.train_id
            data = self.dataset[str(set_num) + '_train']
        elif mode == 'va':
            id = self.valid_id
            data = self.dataset[str(set_num) + '_valid']
        elif mode == 'te':
            id = self.test_id
            data = self.dataset[str(set_num) + '_test']

        batch_size = (batch_size if id + batch_size <= len(data) else len(data) - id)
        padded_data = self.pad_dataset(copy.deepcopy(data[id:id + batch_size]), set_num)
        stories = [d[0] for d in padded_data]
        questions = [d[1] for d in padded_data]
        answers = [d[2] for d in padded_data]
        if len(np.array(answers).shape) < 2:
            for answer in answers:
                while len(answer) != self.config.max_alen:
                    answer.append(-100)
        sup_facts = [d[3] for d in padded_data]
        for sup_fact in sup_facts:
            while len(sup_fact) < self.config.max_episode:
                sup_fact.append(self.config.max_sentnum[set_num] + 1)  # 用max_sentnum+1做padding
        s_lengths = [[idx + 1 for idx, val in enumerate(d[0])
                      if val == self.word2idx['.']] for d in padded_data]
        e_lengths = []
        for s_len in s_lengths:
            e_lengths.append(len(s_len))
            while len(s_len) != self.config.max_sentnum[set_num]:
                s_len.append(0)
        q_lengths = [[idx + 1 for idx, val in enumerate(d[1])
                      if val == self.word2idx['?']][0] for d in padded_data]

        if mode == 'tr':
            self.train_id = (id + batch_size) % len(data)
        elif mode == 'va':
            self.valid_id = (id + batch_size) % len(data)
        elif mode == 'te':
            self.test_id = (id + batch_size) % len(data)
        return (stories, questions, answers, sup_facts,
                s_lengths, q_lengths, e_lengths)

    def get_batch_dataloader(self, mode='tr', set_num=1, batch_size=None):
        """
        获取一个迭代式的dataloader，重写方法DataLoader类
        :param mode:
        :param set_num:
        :param batch_size:
        :return:
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        if mode == 'tr':
            data = self.dataset[str(set_num) + '_train']
        elif mode == 'va':
            data = self.dataset[str(set_num) + '_valid']
        elif mode == 'te':
            data = self.dataset[str(set_num) + '_test']
        padded_data = self.pad_dataset(copy.deepcopy(data), set_num)
        # padded_data： (27000总条数 / 15每个实例的input数 * 5每个实例的question数)
        dset = MyDataset(padded_data, set_num, self.config, self.word2idx)

        def collate_fn(batch):
            """
            复写dataloader batch返回函数
            :param batch:
            :return:
            """
            story = torch.LongTensor([item[0] for item in batch])
            question = torch.LongTensor([item[1] for item in batch])
            answer = torch.LongTensor([item[2] for item in batch])
            sup_fact = torch.LongTensor([[it - 1 for it in item[3]] for item in batch])
            s_len = torch.LongTensor([item[4] for item in batch])
            q_len = torch.LongTensor([item[5] for item in batch])
            e_len = torch.LongTensor([item[6] for item in batch])
            return story, question, answer, sup_fact, s_len, q_len, e_len

        dataloader = DataLoader(dataset=dset,
                                shuffle=True,
                                collate_fn=collate_fn,
                                batch_size=batch_size)
        return dataloader

    def get_dataset_len(self, mode, set_num):
        """
        获取数据集的长度
        :param mode:        数据集模式，train/valid/test
        :param set_num:     数据集序号
        :return:            当前数据集长度
        """
        if mode == 'tr':
            return len(self.dataset[str(set_num) + '_train'])
        elif mode == 'va':
            return len(self.dataset[str(set_num) + '_valid'])
        elif mode == 'te':
            return len(self.dataset[str(set_num) + '_test'])

    def shuffle_data(self, mode='tr', set_num=1, seed=None):
        """
        随机打乱数据
        :param mode:        数据集模式，train/valid/test
        :param set_num:     数据集序号
        :param seed:        随机种子
        :return:
        """
        if seed is not None:
            np.random.seed(seed)
        if mode == 'tr':
            np.random.shuffle(self.dataset[str(set_num) + '_train'])
        elif mode == 'va':
            np.random.shuffle(self.dataset[str(set_num) + '_valid'])
        elif mode == 'te':
            np.random.shuffle(self.dataset[str(set_num) + '_test'])

    def decode_data(self, story, question, answer, support_fact, length_sentences):
        """
        规则化打印数据
        :param story:               输入模块的fact
        :param question:            问句模块的question
        :param answer:              答案模块的answer
        :param support_fact:        答案的支撑fact集合(11是padding，不具有实际意义)
        :param length_sentences:    每个输入句子的单词个数长度
        :return:
        """
        print(length_sentences)
        print('story:',
              ' '.join(self.map_word2idx(story[:length_sentences[-1]], self.idx2word)))
        print('question:', ' '.join(self.map_word2idx(question, self.idx2word)))
        print('answer:', self.map_word2idx(answer, self.idx2word))
        print('supporting fact:', support_fact)
        print('length of sentences:', length_sentences)


if __name__ == '__main__':
    config = Config()
    if config.load_preprocess:
        print('## load preprocess %s' % config.preprocess_path)
        dataset = pickle.load(open(config.preprocess_path, 'rb'))
        dataset.valid_data()
    else:
        dataset = Dataset(config)
        dataset.preprocess()
