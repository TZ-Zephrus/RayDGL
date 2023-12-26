import random
import numpy as np
import numpy.random
import torch
import dgl
import torch.nn as nn
from dgl.data import CoraGraphDataset, PubmedGraphDataset, RedditDataset, FlickrDataset, YelpDataset
from igb.dataloader import IGB260MDGLDataset
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.nn.pytorch import GraphConv, SAGEConv
import argparse
import time


def connect_edge(origin_u, origin_v, del_node):
    time = len(origin_u)  # 记录一共所需的循环次数，也就是最开始的列表长度
    e = 0
    for i in range(time):
        if ((origin_u[i] in del_node) and (origin_v[i] not in del_node)) or ((origin_u[i] not in del_node) and (origin_v[i] in del_node)):
            e+=1
        # print('i = ', i)
    return e



datasetname = 'reddit'
num_parts = 100
if datasetname == 'cora':
    dataset = CoraGraphDataset(raw_dir='/home/asd/文档/wtz/wtz/RayDGL/dataset/{}'.format(datasetname))
    num_class = 7
if datasetname == 'cora_nobalance':
    dataset = CoraGraphDataset(raw_dir='/home/asd/文档/wtz/wtz/RayDGL/dataset/{}'.format(datasetname))
    num_class = 7
if datasetname == 'pubmed':
    dataset = PubmedGraphDataset(raw_dir='/home/asd/文档/wtz/wtz/RayDGL/dataset/{}'.format(datasetname))
    num_class = 3
if datasetname == 'reddit':
    dataset = RedditDataset(raw_dir='/home/asd/文档/wtz/wtz/RayDGL/dataset/{}'.format(datasetname))
    num_class = 41
if datasetname == 'reddit_ballancetrain':
    dataset = RedditDataset(raw_dir='/home/asd/文档/wtz/wtz/RayDGL/dataset/{}'.format('reddit'))
    num_class = 41
if datasetname == 'reddit_ballancelabel':
    dataset = RedditDataset(raw_dir='/home/asd/文档/wtz/wtz/RayDGL/dataset/{}'.format('reddit'))
    num_class = 41
if datasetname == 'flickr':
    dataset = FlickrDataset(raw_dir='/home/asd/文档/wtz/wtz/RayDGL/dataset/{}'.format(datasetname))
    num_class = 7
if datasetname == 'igb_small':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/home/asd/文档/wtz/wtz/RayDGL/dataset/igb_small', 
        help='path containing the datasets')
    parser.add_argument('--dataset_size', type=str, default='small',
        choices=['tiny', 'small', 'medium', 'large', 'full'], 
        help='size of the datasets')
    parser.add_argument('--num_classes', type=int, default=19, 
        choices=[19, 2983], help='number of classes')
    parser.add_argument('--in_memory', type=int, default=1, 
        choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    parser.add_argument('--synthetic', type=int, default=0,
        choices=[0, 1], help='0:nlp-node embeddings, 1:random')
    args2 = parser.parse_args()
    dataset = IGB260MDGLDataset(args2)
    num_class = 19
if datasetname == 'yelp':
    dataset = YelpDataset(raw_dir='/home/asd/文档/wtz/wtz/RayDGL/dataset/{}'.format(datasetname))
    num_class = 100
if datasetname == 'ogbn_products':
    dataset = DglNodePropPredDataset(name = 'ogbn-products', root='/home/asd/文档/wtz/wtz/RayDGL/dataset/ogbn_products')
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph, label = dataset[0] # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
if datasetname == 'ogbn_arxiv':
    dataset = DglNodePropPredDataset(name = 'ogbn-arxiv', root='/home/asd/文档/wtz/wtz/RayDGL/dataset/ogbn_arxiv')
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    graph, label = dataset[0] # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
    
    
if 'ogbn' in datasetname:
    graph_full, label = dataset[0] # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
    edges = graph_full.edges()
    nodes = graph_full.nodes()
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    train_mask = torch.zeros(graph_full.num_nodes(), dtype=bool)
    val_mask = torch.zeros(graph_full.num_nodes(), dtype=bool)
    test_mask = torch.zeros(graph_full.num_nodes(), dtype=bool)
    train_mask[train_idx] = True
    val_mask[valid_idx] = True
    test_mask[test_idx] = True
    label = label.flatten()           # 张量展开为1维
    feat = graph_full.ndata['feat']    
    u = edges[0].numpy()
    v = edges[1].numpy()
    u_list = edges[0].tolist()
    v_list = edges[1].tolist()
else:
    graph_full = dataset[0]
    edges = graph_full.edges()
    nodes = graph_full.nodes()
    train_mask = graph_full.ndata['train_mask'].bool()
    val_mask = graph_full.ndata['val_mask'].bool()
    test_mask = graph_full.ndata['test_mask'].bool()
    label = graph_full.ndata['label']
    feat = graph_full.ndata['feat']
    u = edges[0].numpy()
    v = edges[1].numpy()
    u_list = edges[0].tolist()
    v_list = edges[1].tolist()
    # print(u)
    # print(v)
    
partition_data = []
all_edge = 0
for i in range(num_parts):
    (
        g_local, node_feats, edge_feats, gpb, graph_name, ntypes_list, etypes_list,
    ) = dgl.distributed.load_partition(part_config='/home/asd/文档/wtz/wtz/RayDGL/dataset/{} {} partition/{}.json'.format(datasetname, num_parts, datasetname), part_id=i)

    num_node = sum(g_local.ndata['inner_node'].numpy())
    num_edge = sum(g_local.edata['inner_edge'].numpy())
    print('partition {}: '.format(i), g_local)
    print('partition {}, inner_node = {}, inner_edge = {}'.format(i, num_node, num_edge))
    num_train = sum(node_feats['_N/train_mask'].numpy())
    num_val = sum(node_feats['_N/val_mask'].numpy())
    num_test = sum(node_feats['_N/test_mask'].numpy())
    # 子图顶点
    num_node = sum(g_local.ndata['inner_node'].numpy())
    print(g_local.edata['inner_edge'])
    print(g_local.edata['inner_edge'].size())
    part_node_id = g_local.ndata['_ID'][:num_node]  # 子图中的顶点，也就是要删除的顶点
    # e = connect_edge(u, v, part_node_id)
    all_edge = all_edge + g_local.num_edges()
    partition_data.append([num_node, num_edge, num_train, num_val, num_test])
    # partition_data.append([num_node, num_edge, num_train, num_val, num_test, e])
print(np.array(partition_data))
print('all node number: ', sum([partition_data[i][0] for i in range(num_parts)]))
print('all edge number: ', sum([partition_data[i][1] for i in range(num_parts)]))
print('all train number: ', sum([partition_data[i][2] for i in range(num_parts)]))
print('all valid number: ', sum([partition_data[i][3] for i in range(num_parts)]))
print('all test number: ', sum([partition_data[i][4] for i in range(num_parts)]))