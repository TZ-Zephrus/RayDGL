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
from ogb.nodeproppred import DglNodePropPredDataset


'''
用于找到n跳邻域中的所有顶点, 并返回真实id
'''
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
    
    