#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
  @ Time     : 2020/7/18 下午6:40
  @ Author   : Vodka
  @ File     : main .py
  @ Software : PyCharm
"""
import argparse
import os
import pickle
import warnings

import torch

from config import Config
from dataset import Dataset
from model import DMN
from run import run_experiment

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    """
    程序参数列表：
        data_path       预处理过的数据路径
        model_name      模型名称
        checkpoint_dir  断点路径
        batch_size      批处理大小
        epoch           迭代次数
        train           是否训练
        valid           是否验证
        test            是否测试
        early_stop      是否提前停止
        resume          是否从断点唤醒
        save            是否保存模型
        print_step      每多少步打印一次
    """
    argparser.add_argument('--data_path', type=str, default='./data/babi(tmp).pkl')
    argparser.add_argument('--model_name', type=str, default='m')
    argparser.add_argument('--checkpoint_dir', type=str, default='./results/')
    argparser.add_argument('--load_model_path', type=str, default="./results/model.pth")
    argparser.add_argument('--save_model_path', type=str, default="./results/model.pth")
    argparser.add_argument('--batch_size', type=int, default=32)
    argparser.add_argument('--epoch', type=int, default=100)
    argparser.add_argument('--train', type=int, default=1)
    argparser.add_argument('--valid', type=int, default=1)
    argparser.add_argument('--test', type=int, default=1)
    argparser.add_argument('--early_stop', type=int, default=0)
    argparser.add_argument('--resume', action='store_true', default=False)
    argparser.add_argument('--save', action='store_true', default=False)
    argparser.add_argument('--print_step', type=float, default=128)
    argparser.add_argument('--device', default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    """
    模型超参数列表：
        lr              学习率
        lr_decay        学习率衰减因子
        wd              权重参数衰减因子
        grad_max_norm   最大梯度剪裁阈值，当梯度小于/大于阈值时，更新的梯度为阈值，为的是避免梯度消失/爆炸
        s_rnn_hdim      input模块 隐层大小
        s_rnn_ln        input模块 一次处理数据个数
        s_rnn_dr        input模块 dropout几率
        q_rnn_hdim      question模块 隐层大小
        q_rnn_ln        question模块 一次处理数据个数
        q_rnn_dr        question模块 dropout几率
        e_cell_hdim     episodic_memory模块 AttentionGRU单元隐层大小
        m_cell_hdim     最终输出memory的 GRU单元隐层大小
        a_cell_hdim     answer模块 GRU单元隐层大小
        word_dr         word2vec_embedding_layer dropout几率
        g1_dim          线性层g1的输出维度
        max_episode     episodic_memory模块 最大迭代次数
        beta_cnt        train样例最大个数
        set_num         文章集序号(1~20)
        max_alen        answer模块 答案的最大长度
    """
    argparser.add_argument('--lr', type=float, default=0.0003)
    argparser.add_argument('--lr_decay', type=float, default=1.0)
    argparser.add_argument('--wd', type=float, default=0)
    argparser.add_argument('--grad_max_norm', type=int, default=5)
    argparser.add_argument('--s_rnn_hdim', type=int, default=100)
    argparser.add_argument('--s_rnn_ln', type=int, default=1)
    argparser.add_argument('--s_rnn_dr', type=float, default=0.0)
    argparser.add_argument('--q_rnn_hdim', type=int, default=100)
    argparser.add_argument('--q_rnn_ln', type=int, default=1)
    argparser.add_argument('--q_rnn_dr', type=float, default=0.0)
    argparser.add_argument('--e_cell_hdim', type=int, default=100)
    argparser.add_argument('--m_cell_hdim', type=int, default=100)
    argparser.add_argument('--a_cell_hdim', type=int, default=100)
    argparser.add_argument('--word_dr', type=float, default=0.2)
    argparser.add_argument('--g1_dim', type=int, default=500)
    argparser.add_argument('--max_episode', type=int, default=10)
    argparser.add_argument('--beta_cnt', type=int, default=10)
    argparser.add_argument('--set_num', type=int, default=1)
    argparser.add_argument('--max_alen', type=int, default=2)
    args = argparser.parse_args()

    if not os.path.exists('./results'):
        os.makedirs('./results')

    print('### load dataset')
    config = Config()
    dataset = Dataset(config)
    dataset = pickle.load(open(args.data_path, 'rb'))
    dataset.config.__dict__.update(args.__dict__)
    args.__dict__.update(dataset.config.__dict__)
    # import pprint
    # pp = lambda x: pprint.PrettyPrinter().pprint(x)
    # pp(args.__dict__)

    # for set_num in range(args.set_num, min(args.set_num + 1, 21)):  # set_num区间：1~20
    for set_num in range(1, 21):  # set_num区间：1~20
        # print('\n[QA set %d]' % (set_num))
        model = DMN(args, dataset.idx2vec, set_num).to(args.device)
        run_experiment(model, dataset, set_num, args.device)

    print('### end of experiment')
