import numpy as np
import argparse
import numpy.random
import torch
import dgl
import torch.nn as nn
from dgl.data import CoraGraphDataset, PubmedGraphDataset, RedditDataset, FlickrDataset, YelpDataset
from dgl.nn.pytorch import GraphConv, SAGEConv
import time
import networkx as nx
import random
import pandas as pd
from igb.dataloader import IGB260MDGLDataset

import argparse
import os
import sys
import time
import dgl
import numpy as np
import torch
from dgl.data import CoraGraphDataset, PubmedGraphDataset, RedditDataset
from igb import download
import argparse, dgl
from igb.dataloader import IGB260MDGLDataset
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt



datasetname = 'cora'
num_parts = 100
GNNlayer = 2

part_id = 98
(
    g_local, node_feats, edge_feats, gpb, graph_name, ntypes_list, etypes_list,
) = dgl.distributed.load_partition(part_config='/home/asd/文档/wtz/wtz/RayDGL/dataset/{} {} partition/{}.json'.format(datasetname, num_parts, datasetname), part_id=part_id)

num_node = sum(g_local.ndata['inner_node'].numpy())
print('partition {}, inner_num_node = {}'.format(part_id, num_node))
# 子图里的点和边
localNodeId = g_local.ndata['_ID'][:num_node]  # local_data 子图中的顶点
redundantNodeId = g_local.ndata['_ID']
localEdgeId = g_local.edata['_ID'][g_local.edata['inner_edge']]
redundantEdgeID = g_local.edata['_ID']
# 
print(g_local.num_nodes())



if datasetname == 'cora':
    dataset = CoraGraphDataset(raw_dir='/home/asd/文档/wtz/wtz/RayDGL/dataset/{}'.format(datasetname))
    num_class = 7
    graph = dataset[0]
if datasetname == 'pubmed':
    dataset = PubmedGraphDataset(raw_dir='/home/asd/文档/wtz/wtz/RayDGL/dataset/{}'.format(datasetname))
    num_class = 3
    graph = dataset[0]
if datasetname == 'reddit':
    dataset = RedditDataset(raw_dir='/home/asd/文档/wtz/wtz/RayDGL/dataset/{}'.format(datasetname))
    num_class = 41
    graph = dataset[0]


# new_graph = dgl.sampling.sample_neighbors(g_local, [15], fanout=-1, edge_dir='out')
# print(new_graph)
def findDegreeNerghbor(graph, neighborNode, hop_range):
    neighborNode = np.array(neighborNode) - graph.ndata['_ID'][graph.ndata['inner_node']][0].tolist()     # 将传过来的原node id转换为子图中的较小的node id
    for i in range(hop_range):
        new_graph = dgl.sampling.sample_neighbors(graph, neighborNode, fanout=-1, edge_dir='out')
        neighborNode = torch.unique(torch.cat((new_graph.edges()[0], new_graph.edges()[1]), dim=0))   # 找到一个跳中的所有顶点   将出点和入点合并后删除重复元素
    # 删除不属于inner_node中的顶点
    neighborNode = neighborNode + graph.ndata['_ID'][graph.ndata['inner_node']][0].tolist()    # 恢复成真实ID
    realMask = (graph.ndata['_ID'][graph.ndata['inner_node']][0].tolist() <= neighborNode).tolist() and (neighborNode <= graph.ndata['_ID'][graph.ndata['inner_node']][-1].tolist()).tolist()
    neighborNode = neighborNode[realMask]
    return neighborNode.tolist()
# fig, ax = plt.subplots()
# nx.draw(g_local.to_networkx(), with_labels=True, ax=ax)   # 将图转为networkx形式
# ax.set_title('Graph')
# plt.savefig('/home/asd/文档/wtz/wtz/figure.png')

print(findDegreeNerghbor(g_local, [2652,2653,2654,2656], 2))