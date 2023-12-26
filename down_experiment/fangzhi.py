import numpy as np
import argparse
import numpy.random
import torch
import dgl
import torch.nn as nn
from dgl.data import CoraGraphDataset, PubmedGraphDataset, RedditDataset, FlickrDataset
from dgl.nn.pytorch import GraphConv, SAGEConv
import time
import networkx as nx
import random
import pandas as pd
from igb.dataloader import IGB260MDGLDataset


datasetname = 'cora'
num_parts = 100
for i in range(100):
    (
        g_local, node_feats, edge_feats, gpb, graph_name, ntypes_list, etypes_list,
    ) = dgl.distributed.load_partition(part_config='/home/asd/文档/wtz/wtz/RayDGL/dataset/{} {} partition/{}.json'.format(datasetname, num_parts, datasetname), part_id=i)

    num_node = sum(g_local.ndata['inner_node'].numpy())
    print('partition {}, num_node = {}'.format(i, num_node))
    part_node_id = g_local.ndata['_ID'][:num_node]  # 子图中的顶点，也就是要删除的顶点
    part_node_id_total = np.append(part_node_id_total, part_node_id)
    part_node_id_total = part_node_id_total.astype(int)