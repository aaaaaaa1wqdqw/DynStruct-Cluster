from __future__ import print_function, division
import argparse
import random
import time

import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
from torch.optim import Adam
from AdaptiveGraphLearning import AdaptiveGraphLearning
from model import DyCluster
from utils import load_data, load_graph
from evaluation import eva, print_best_metrics

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def train_sdcn(dataset):
    set_seed(args.seed)
    model = DyCluster(500, 500, 2000, 2000, 500, 500,
                 n_input=args.n_input,
                 n_z=args.n_z,
                 n_clusters=args.n_clusters,
                 v=1.0,
                 pretrain_path = args.pretrain_path).to(device)

    adaptive_graph = AdaptiveGraphLearning(k=args.topk, lambda_init=args.l, learn_lambda=True).to(device)
    optimizer = Adam(list(model.parameters()) + list(adaptive_graph.parameters()), lr=args.lr)

    adj = load_graph(args.name, args.k)
    adj = adj.cuda()

    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y
    with torch.no_grad():
        x_bar, _, _, _, z = model.ae(data)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans. cluster_centers_).to(device)
    pseudo_lablles = torch.tensor(y_pred).to(device)

    for epoch in range(args.epoch):
        A_final, A_learned = adaptive_graph(data, adj)
        x_bar, q, pred, z, h = model(data, A_final,args.sigma)
        if epoch == 0:
            pseudo_labels = pred.argmax(1)
            model.initialize_prototypes(F.normalize(h, p=2, dim=1), pseudo_labels)
        else:
            with torch.no_grad():
                pseudo_labels = pred.argmax(1)
                model.update_prototypes(F.normalize(h, p=2, dim=1), pseudo_labels)
        res2 = pred.detach().cpu().numpy().argmax(1)
        acc, nmi, ari, f1=eva(y, res2, str(epoch))
        p = target_distribution(q.detach())
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
        re_loss = F.mse_loss(x_bar, data)
        proto_loss = model.prototype_loss(F.normalize(h, p=2, dim=1), pseudo_labels)
        graph_reg_loss = (A_learned ** 2).sum()
        loss = 0.1 * kl_loss + 0.01 * ce_loss + re_loss + args.proto_weight * proto_loss + args.graph_reg_weight * graph_reg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print_best_metrics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='dblp')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n_clusters', default=3, type=int)
    parser.add_argument('--n_z', default=10, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')

    parser.add_argument('--temperature', type=float, default=0.1,
                        help='Temperature parameter for prototype contrastive learning')

    parser.add_argument('--proto_weight', type=float, default=0.5,help='Weight for prototype contrastive loss')
    parser.add_argument('--graph_reg_weight', type=float, default=1e-3, help='Weight for graph regularization term')
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--l', type=float, default=0.5)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--sigma', type=int, default=0.5)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    args.pretrain_path = 'data/{}.pkl'.format(args.name)
    dataset = load_data(args.name)

    if args.name == 'hhar':
        args.k = 5
        args.n_clusters = 6
        args.n_input = 561

    if args.name == 'reut':
        args.lr = 1e-4
        args.n_clusters = 4
        args.n_input = 2000

    if args.name == 'acm':
        args.k = None
        args.n_clusters = 3
        args.n_input = 1870

    if args.name == 'dblp':
        args.k = None
        args.n_clusters = 4
        args.n_input = 334
        args.seed = 8644
        args.topk = 50
        args.sigma = 0.1
    if args.name == 'cite':
        args.lr = 1e-4
        args.k = None
        args.n_clusters = 6
        args.n_input = 3703


    start_time = time.time()
    train_sdcn(dataset)
    end_time = time.time()
    print(f"Total Time: {end_time - start_time:.2f} seconds")