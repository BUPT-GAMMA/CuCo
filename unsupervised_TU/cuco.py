import os.path as osp
import scipy.stats as stats
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import pandas as pd

import visualization
from aug import TUDataset_aug as TUDataset
from torch_geometric.data import DataLoader
import sys
import json
from torch import optim

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *
from gin import Encoder
from evaluate_embedding import evaluate_embedding
from model import *
import time
from arguments import arg_parse

import torch.utils.data as utils

from torch_geometric.transforms import Constant
import pdb


class GcnInfomax(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
        super(GcnInfomax, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # self.prior = args.prior

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

        self.local_d = FF(self.embedding_dim)
        self.global_d = FF(self.embedding_dim)
        # self.local_d = MI1x1ConvNet(self.embedding_dim, mi_units)
        # self.global_d = MIFCNet(self.embedding_dim, mi_units)

        # if self.prior:
        #     self.prior_d = PriorDiscriminator(self.embedding_dim)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, num_graphs):

        # batch_size = data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        y, M = self.encoder(x, edge_index, batch)

        g_enc = self.global_d(y)
        l_enc = self.local_d(M)

        mode = 'fd'
        measure = 'JSD'
        local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)

        if self.prior:
            prior = torch.rand_like(y)
            term_a = torch.log(self.prior_d(prior)).mean()
            term_b = torch.log(1.0 - self.prior_d(y)).mean()
            PRIOR = - (term_a + term_b) * self.gamma
        else:
            PRIOR = 0

        return local_global_loss + PRIOR


class simclr(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, dataset_num_features, dataset_num, alpha=0.5, beta=1., gamma=.1):
        super(simclr, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.dataset_num = dataset_num
        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    @torch.no_grad()
    def sample_negative_index(self, negative_number, epoch, epochs):

        lamda = 1/2
        # lamda = 1
        #lamda = 2
        lower, upper = 0, self.dataset_num
        mu_1 = ((epoch-1) / epochs) ** lamda * (upper - lower)
        mu_2 = ((epoch) / epochs) ** lamda * (upper - lower)
        # sigma = negative_number / 6
        # # X表示含有最大最小值约束的正态分布
        # X = stats.truncnorm(
        #     (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)  # 有区间限制的随机数 正态分布采样
        # X = stats.uniform(mu_1,mu_2-mu_1)  # 均匀分布采样
        X = stats.uniform(1,mu_2)
        index = X.rvs(negative_number)  # 采样
        index = index.astype(np.int)
        return index

    def forward(self, x, edge_index, batch, num_graphs):

        # batch_size = data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        y, M = self.encoder(x, edge_index, batch)

        y = self.proj_head(y)

        return y

    def rank_negative_queue(self, x1, x2):

        x2 = x2.t()
        x = x1.mm(x2)

        x1_frobenius = x1.norm(dim=1).unsqueeze(0).t()
        x2_frobenins = x2.norm(dim=0).unsqueeze(0)
        x_frobenins = x1_frobenius.mm(x2_frobenins)

        final_value = x.mul(1 / x_frobenins)

        sort_queue, _ = torch.sort(final_value, dim=0, descending=False)

        return sort_queue

    def loss_cal(self, q_batch, q_aug_batch, negative_sim):

        T = 0.2

        # q_batch = q_batch[: q_aug_batch.size()[0]]

        positive_sim = torch.cosine_similarity(q_batch, q_aug_batch, dim=1)  # 维度有时对不齐

        positive_exp = torch.exp(positive_sim / T)

        negative_exp = torch.exp(negative_sim / T)

        negative_sum = torch.sum(negative_exp, dim=0)

        loss = positive_exp / (positive_exp+negative_sum)

        loss = -torch.log(loss).mean()

        return loss


import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':

    args = arg_parse()
    setup_seed(args.seed)

    accuracies = {'val': [], 'test': []}
    log_loss = [float("inf")]
    epochs = 20
    log_interval = 1
    batch_size = args.batch_size
    # batch_size = 512
    lr = args.lr
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

    dataset = TUDataset(path, name=DS, aug=args.aug).shuffle()
    dataset_eval = TUDataset(path, name=DS, aug='none').shuffle()
    dataset_num = len(dataset)

    # if args.neg_num < dataset_num:
    #     negative_number = args.neg_num
    # else:
    #     negative_number = dataset_num

    # negative_number = int(dataset_num/epochs)
    negative_number = 16

    print(dataset_num)
    print(dataset.get_num_feature())
    print(negative_number)

    try:
        dataset_num_features = dataset.get_num_feature()
    except:
        dataset_num_features = 1

    dataloader = DataLoader(dataset, batch_size=batch_size)
    dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = simclr(args.hidden_dim, args.num_gc_layers, dataset_num_features, dataset_num).to(device)
    # print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('================')

    model.eval()
    emb, y = model.encoder.get_embeddings(dataloader_eval)
    # print(emb.shape, y.shape)

    """
    acc_val, acc = evaluate_embedding(emb, y)
    accuracies['val'].append(acc_val)
    accuracies['test'].append(acc)
    """
    stop_counter = 0
    patience = 3

    for epoch in range(1, epochs+1):
        loss_all = 0
        start_time = time.time()
        model.train()

        # if (epoch == 1) or (epoch % 5 == 0):  # 定期更新样本的表示
        dataset_embedding, _ = model.encoder.get_embeddings(dataloader)
        dataset_embedding = torch.from_numpy(dataset_embedding).to(device)

        for data in dataloader:

            data, data_aug = data
            optimizer.zero_grad()

            node_num, _ = data.x.size()

            data = data.to(device)

            q_batch = model(data.x, data.edge_index, data.batch, data.num_graphs)

            if args.aug == 'dnodes' or args.aug == 'subgraph' or args.aug == 'random2' or args.aug == 'random3' or args.aug == 'random4':
                # node_num_aug, _ = data_aug.x.size()
                edge_idx = data_aug.edge_index.numpy()
                _, edge_num = edge_idx.shape
                idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]

                node_num_aug = len(idx_not_missing)
                data_aug.x = data_aug.x[idx_not_missing]

                data_aug.batch = data.batch[idx_not_missing]
                idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}
                edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if
                            not edge_idx[0, n] == edge_idx[1, n]]
                data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)

            data_aug = data_aug.to(device)

            q_aug_batch = model(data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs)

            # print(data.edge_index)
            # print(data.edge_index.size())
            # print(data_aug.edge_index)
            # print(data_aug.edge_index.size())
            # print(data.x.size())
            # print(data_aug.x.size())
            # print(data.batch.size())
            # print(data_aug.batch.size())
            # pdb.set_trace()

            # print(q_batch.size())
            # print(q_aug_batch.size())

            q_batch = q_batch[: q_aug_batch.size()[0]]

            sort_queue = model.rank_negative_queue(dataset_embedding, q_batch)

            sample_index = model.sample_negative_index(negative_number, epoch, epochs)

            sample_index = torch.tensor(sample_index).to(device)

            negative_sim = sort_queue.index_select(0, sample_index)

            loss = model.loss_cal(q_batch, q_aug_batch, negative_sim)
            loss_all += loss.item()
            loss.backward()
            optimizer.step()

        end_time = time.time()
        print('Epoch {}, Loss {}'.format(epoch, loss_all))
        print('time: {} s'.format(end_time - start_time))

        log_loss.append(loss_all)

        if log_loss[-1] > log_loss[-2]:  # early stop
            stop_counter += 1
            negative_number = int(negative_number/2)
            if stop_counter > patience or negative_number <= 2:
                model.eval()
                emb, y = model.encoder.get_embeddings(dataloader_eval)
                # visualization.draw_plot(DS=args.DS, embeddings=emb, labels=y)          #可视化
                # pdb.set_trace()
                acc_val, acc = evaluate_embedding(emb, y)
                accuracies['val'].append(acc_val)
                accuracies['test'].append(acc)
                # accuracies['loss'].append(loss_all / len(dataloader))
                break

        if epoch % log_interval == 0:
            model.eval()
            emb, y = model.encoder.get_embeddings(dataloader_eval)
            acc_val, acc = evaluate_embedding(emb, y)
            accuracies['val'].append(acc_val)
            accuracies['test'].append(acc)
            # accuracies['loss'].append(loss_all/len(dataloader))
        # print(accuracies['val'][-1], accuracies['test'][-1])

    tpe = ('local' if args.local else '') + ('prior' if args.prior else '')

    pd.DataFrame(accuracies).to_csv('Result/' + args.DS + '_result.csv', index=False)

    with open('logs/log_' + args.DS + '_' + args.aug, 'a+') as f:
        s = json.dumps(accuracies)
        f.write('{},{},{},{},{},{},{},{},{}\n'.format(args.DS, tpe, args.num_gc_layers, epochs, log_interval, lr,
                                                      args.batch_size, negative_number, s))
        f.write('\n')
