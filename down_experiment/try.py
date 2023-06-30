import argparse
import os
import sys
import time
import dgl
import numpy as np
import torch as th
from dgl.data import CoraGraphDataset
np.set_printoptions(threshold=np.inf)


# dataset = CoraGraphDataset(raw_dir='D:/1a一些学习/一些代码/graph NN/dataset/cora')
# g=dataset[0]
#
# (
#     g_local, node_feats, edge_feats, gpb, graph_name, ntypes_list, etypes_list,
# ) = dgl.distributed.load_partition(part_config='D:/1a一些学习/一些代码/graph NN/dataset/cora 3 partition/cora.json', part_id=0)
#
#
# print(sum(g_local.ndata['inner_node'].numpy()))

a = np.array([0,0,0,1,2,3,3,4,5])
b = np.array([1,2,5,2,3,5,4,3,0])
c = np.array([1,5])
print(a[:5])

# def del_edge(origin_u, origin_v, del_node):
#     time = len(origin_u)  # 记录一共所需的循环次数，也就是最开始的列表长度
#     j = 0
#     for i in range(time):
#         if origin_u[j] in del_node or origin_v[j] in del_node:
#             origin_u = np.delete(origin_u, j)
#             origin_v = np.delete(origin_v, j)
#             j-=1
#         j+=1
#     return origin_u, origin_v
# d,e = del_edge(a,b,c)
# print(d,e)

# a = np.array([0,0,0,1,2,3,3,4,5])
# print(len(a))