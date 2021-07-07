from aug import TUDataset_aug as TUDataset
from torch_geometric.data import DataLoader
from arguments import arg_parse
import os.path as osp
import torch
import gin
import torch.nn.functional as F
import scipy.stats as stats
import numpy as np

from gsimclr import *

torch.cuda.set_device(2)


def NCEloss(q, q_aug, negtive_sample):
    T = 0.2

    pos_sim = abs(F.cosine_similarity(q, q_aug, dim=0))
    positive_exp = torch.exp(pos_sim / T)

    negative_exp = 0
    for n_sample in negtive_sample:
        neg_sim = abs(F.cosine_similarity(q, n_sample, dim=0))
        negative_exp = negative_exp + torch.exp(neg_sim / T)

    NCELoss = -torch.log(positive_exp / (positive_exp + negative_exp))
    return NCELoss


def sample_negative(dataset_number, negative_number, epoch, epochs, negative_queue):
    lower, upper = 0, dataset_number
    mu = epoch / epochs * negative_number
    sigma = (1 / (3 * epochs)) * negative_number
    # X表示含有最大最小值约束的正态分布
    X = stats.truncnorm(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)  # 有区间限制的随机数
    index = X.rvs(negative_number)  # 采样
    index = index.astype(np.int)
    sample_result = []
    for i in range(0, negative_number):
        sample_result.append(negative_queue[index[i]])

    sample_result = torch.stack(sample_result).to(device)

    return sample_result




path = osp.join(osp.dirname(osp.realpath(__file__)), '../..', 'data', "MUTAG")

dataset = TUDataset(path, name='MUTAG', aug='random2').shuffle()
dataset_eval = TUDataset(path, name=DS, aug='none').shuffle()

print(len(dataset_eval))
print(len(dataset))
print(dataset.get_num_feature())

try:
    dataset_num_features = dataset.get_num_feature()
except:
    dataset_num_features = 1

args = arg_parse()

batch_size = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = simclr(32, 3, dataset_num_features, len(dataset)).to(device)

import time

start_time = time.time()

dataloader = DataLoader(dataset, batch_size=batch_size)

op_time = time.time()

print('cal_dataloader {}'.format(op_time-start_time))

K = len(dataset)

metric = 'cosine'
args = arg_parse()
setup_seed(args.seed)
num_dim = 96

negative_number = 64

epoch = 1
epochs = 20
#
# un_queue = torch.randn(96).to(device)

# start_time = time.time()
import time

start_time = time.time()

dataset_embedding, _ = model.encoder.get_embeddings(dataloader)  # 得到所有样本的表示
dataset_embedding = torch.from_numpy(dataset_embedding).to(device)

op_time = time.time()
print('cal_embedding {}'.format(op_time-start_time))
negative_dict = {}


for data in dataloader:

    data, data_aug = data

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

    q_aug = data_aug.to(device)

    q_aug_batch = model(data_aug.x, data_aug.edge_index, data_aug.batch, data_aug.num_graphs)

    start_time = time.time()

    negative_number = 128

    sort_queue = model.rank_negative_queue(dataset_embedding, q_batch)

    sample_index = model.sample_negative_index(negative_number, epoch, epochs)

    sample_index = torch.tensor(sample_index).to(device)

    negative_sim = sort_queue.index_select(0, sample_index)

    loss = model.loss_cal(q_batch, q_aug_batch, negative_sim)

    print(loss)



    # for q, q_aug in zip(q_batch, q_aug_batch):  # to do  batch乘法 代替
    #
    #     print(q)
    #     print(q_aug)
    #
    #     if (q not in negative_dict) or (epoch % 5 == 0):     # 定期更新样本队列
    #
    #         sort_sample_queue = model.rank_function(q, dataset_embedding, "cosine", device)  # to do，优化, batch 矩阵乘法 ！！！
    #         negative_dict[q] = sort_sample_queue
    #
    #     sample_result = model.sample_negative(negative_number, epoch, epochs, negative_dict[q], device)
    #
    #     loss = NCEloss(q, q_aug, sample_result)




    #     losses = losses + loss
    #
    # op_time = time.time()
    #
    # print('cal_batch {}'.format(op_time - start_time))
    #
    # losses = losses.mean()



    # end_time = time.time()
    #
    # print(end_time-start_time)
    #
    # print(1)


