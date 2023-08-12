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
这里用出度来衡量顶点重要性
'''
def arrangeOutDegree(graph, ratio):
    ratio = ratio  # 保留率
    # 将原来g_local的出边edges中不属于inner_edges的边全部删除
    realEdges = graph.edges()[0].numpy() + graph.ndata['_ID'][graph.ndata['inner_node']][0].tolist()
    realMask = (graph.ndata['_ID'][graph.ndata['inner_node']][0].tolist() <= realEdges).tolist() and (realEdges <= graph.ndata['_ID'][graph.ndata['inner_node']][-1].tolist()).tolist()
    realEdges = realEdges[realMask]
    '''
    以上, realEdges中的所有出边, 均是从inner_node中出
    '''
    # 使用Counter统计出边中，每个顶点出现的次数
    counter1 = Counter(realEdges)
        # counter2 = Counter(g_local.edges()[1].tolist())
        # # 要查找的特定元素
        # target_element = 2700
        # # 获取特定元素出现的次数
        # count_of_target1 = counter1[target_element]
        # # count_of_target2 = counter2[target_element]
        # print(count_of_target1)
    # 只有出度在前num_node*0.1名的才会被冗余存下来
    sorted_elements = counter1.most_common(n = int(num_node*ratio))
    # print(sorted_elements)
    sorted_elements_only = [element for element, count in sorted_elements]
    # print(sorted_elements_only)  # 这里出现的，是出度排名靠前的顶点，这些顶点将不会被删去。
    return sorted_elements_only

if __name__ == '__main__':
    datasetname = 'reddit'
    num_parts = 100
    GNNlayer = 2

    part_id = 99
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
    # print(g_local.num_nodes())


    '''
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
    if datasetname == 'ogbn_products':
        dataset = DglNodePropPredDataset(name = 'ogbn-products', root='/home/asd/文档/wtz/wtz/RayDGL/dataset/ogbn_products')
        num_class = 47
        
        graph, label = dataset[0] # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
        edges = graph.edges()
        nodes = graph.nodes()
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        train_mask = torch.zeros(graph.num_nodes(), dtype=bool)
        val_mask = torch.zeros(graph.num_nodes(), dtype=bool)
        test_mask = torch.zeros(graph.num_nodes(), dtype=bool)
        train_mask[train_idx] = True
        val_mask[valid_idx] = True
        test_mask[test_idx] = True
        label = label.flatten()           # 张量展开为1维
        feat = graph.ndata['feat']    
        u = edges[0].numpy()
        v = edges[1].numpy()
        u_list = edges[0].tolist()
        v_list = edges[1].tolist()
    '''
    outDegreeList = arrangeOutDegree(g_local, ratio=0.1)
    print(outDegreeList)

    


   

'''
都是双向边？？？？？？？？？？？
'''