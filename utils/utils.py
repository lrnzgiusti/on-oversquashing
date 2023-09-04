#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: On Oversquashing project authors 
"""

import os
import torch
import numpy as np
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from functorch import vmap
from torch.utils.data.sampler import  Sampler
import random 


eps_min = 1e-12
eps_max = 1e+12

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def coo2tensor(A):
    assert(sp.isspmatrix_coo(A))
    idxs = torch.LongTensor(np.vstack((A.row, A.col)))
    vals = torch.FloatTensor(A.data)
    return torch.sparse_coo_tensor(idxs, vals, size = A.shape, requires_grad = False)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def degree_normalize_adjacency(A):
    """
    NormalizeAdjacency: Computes the degree-normalized adjacency matrix
    Input:
        A (np.array): adjacency matrix
    Output:
        D^-1 • A  • D^-1(sp.coo): degree-normalized adjacency matrix (sparse)
    """
    assert A.shape[0] == A.shape[1]
    d = np.sum(A, axis = 1)
    d = 1/np.sqrt(d)
    D = np.diag(d)
    return sp.coo_matrix(D @ A @ D)

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and
    conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, placeholders):
    """Construct feed dictionary for GCN-Align."""
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] \
                      for i in range(len(support))})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k.
    Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def loadfile(fn, num=1):
    """Load a file and return a list of tuple containing $num integers in each line."""
    print('loading a file...' + fn)
    ret = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split('\t')
            x = []
            for i in range(num):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret


def get_ent2id(fns):
    ent2id = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                ent2id[th[1]] = int(th[0])
    return ent2id


def loadattr(fns, e, ent2id):
    """The most frequent attributes are selected to save space."""
    cnt = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] not in ent2id:
                    continue
                for i in range(1, len(th)):
                    if th[i] not in cnt:
                        cnt[th[i]] = 1
                    else:
                        cnt[th[i]] += 1
    fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]
    num_features = min(len(fre), 2000)
    attr2id = {}
    for i in range(num_features):
        attr2id[fre[i][0]] = i
    M = {}
    for fn in fns:
        with open(fn, 'r', encoding='utf-8') as f:
            for line in f:
                th = line[:-1].split('\t')
                if th[0] in ent2id:
                    for i in range(1, len(th)):
                        if th[i] in attr2id:
                            M[(ent2id[th[0]], attr2id[th[i]])] = 1.0
    row = []
    col = []
    data = []
    for key in M:
        row.append(key[0])
        col.append(key[1])
        data.append(M[key])
    return sp.coo_matrix((data, (row, col)), shape=(e, num_features)) # attr


def get_dic_list(e, KG):
    M = {}
    for tri in KG:
        if tri[0] == tri[2]:
            continue
        M[(tri[0], tri[2])] = 1
        M[(tri[2], tri[0])] = 1
    dic_list = {}
    for i in range(e):
        dic_list[i] = []
    for pair in M:
        dic_list[pair[0]].append(pair[1])
    return dic_list


def func(KG):
    head = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            head[tri[1]] = set([tri[0]])
        else:
            cnt[tri[1]] += 1
            head[tri[1]].add(tri[0])
    r2f = {}
    for r in cnt:
        r2f[r] = len(head[r]) / cnt[r]
    return r2f


def ifunc(KG):
    tail = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            tail[tri[1]] = set([tri[2]])
        else:
            cnt[tri[1]] += 1
            tail[tri[1]].add(tri[2])
    r2if = {}
    for r in cnt:
        r2if[r] = len(tail[r]) / cnt[r]
    return r2if


def get_weighted_adj(e, KG):
    r2f = func(KG)
    r2if = ifunc(KG)
    M = {}
    for tri in KG:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = max(r2if[tri[1]], 0.3)
        else:
            M[(tri[0], tri[2])] += max(r2if[tri[1]], 0.3)
        if (tri[2], tri[0]) not in M:
            M[(tri[2], tri[0])] = max(r2f[tri[1]], 0.3)
        else:
            M[(tri[2], tri[0])] += max(r2f[tri[1]], 0.3)
    row = []
    col = []
    data = []
    for key in M:
        row.append(key[1])
        col.append(key[0])
        data.append(M[key])
    return sp.coo_matrix((data, (row, col)), shape=(e, e))


def get_ae_input(attr):
    return sparse_to_tuple(sp.coo_matrix(attr))

def set_seed(seed):
    # Set the seed for everything
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]

def namespace_to_command(namespace: str, exp_file: str) -> str:
    namespace = namespace.replace("Namespace(", "").replace(")", "")
    args_split = namespace.split(", ")
    
    args = ""
    filter_out = ["device", "input_dim", "output_dim", "edge_dim", "sha", "index"]
    for arg in args_split:
        if any(substring in arg for substring in filter_out):
            continue
        name, value = arg.split("=")
        arg_formatted = f"--{name} {value}"
        args += f"{arg_formatted} "
    return f"python exp/{exp_file} {args}"

class CustomSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Args:
        indices (sequence): a sequence of indices
        train (bool): if the sequence has to be shuffles
    """

    def __init__(self, indices : list, train : bool) -> None:
        self.indices = indices
        self.train = train
        
    def __iter__(self):
      if self.train:
        for i in torch.randperm(len(self.indices)):
            yield self.indices[i]
      else:
        for i in self.indices:
            yield i

    def __len__(self) -> int:
        return len(self.indices)

def smooth_plot(x, y=None, ax=None, label='', halflife=10):

    """ 
    Function to plot smoothed x VS y graphs.

    Parameters: 
    ----------

    x - x-axis data.
    y - y-axis data.
    label - Legend for the current graph.
    halflife - Smoothing level.

    ----------

    Yields: x VS y graphs.

    ----------
    """

    if y is None:
      y_int = x
    else:
      y_int = y
    
    x_ewm = pd.Series(y_int).ewm(halflife=halflife)
    color = next(plt.gca()._get_lines.prop_cycler)['color']
    if y is None:
      ax.plot(x_ewm.mean(), label=label, color=color)
      ax.fill_between(np.arange(x_ewm.mean().shape[0]), x_ewm.mean() + x_ewm.std(), x_ewm.mean() - x_ewm.std(), color=color, alpha=0.15)
    else:
      ax.plot(x, x_ewm.mean(), label=label, color=color)
      ax.fill_between(x, y_int + x_ewm.std(), y_int - x_ewm.std(), color=color, alpha=0.15)
