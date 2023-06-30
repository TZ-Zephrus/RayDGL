import random
import numpy as np
import numpy.random
import torch
import dgl
import torch.nn as nn
from dgl.data import CoraGraphDataset, PubmedGraphDataset, RedditDataset
from dgl.nn.pytorch import GraphConv, SAGEConv
import time

def connect_edge(origin_u, origin_v, del_node):
    time = len(origin_u)  # 记录一共所需的循环次数，也就是最开始的列表长度
    e = 0
    for i in range(time):
        if ((origin_u[i] in del_node) and (origin_v[i] not in del_node)) or ((origin_u[i] not in del_node) and (origin_v[i] in del_node)):
            e+=1
        # print('i = ', i)
    return e

datasetname = 'pubmed'
num_parts = 100

if datasetname == 'cora':
    dataset = CoraGraphDataset(raw_dir='/root/wtz/RayDGL/dataset/{}'.format(datasetname))
    num_class = 7
if datasetname == 'pubmed':
    dataset = PubmedGraphDataset(raw_dir='/root/wtz/RayDGL/dataset/{}'.format(datasetname))
    num_class = 3
if datasetname == 'reddit':
    dataset = RedditDataset(raw_dir='/root/wtz/RayDGL/dataset/{}'.format(datasetname))
    num_class = 41
graph = dataset[0]
edges = graph.edges()
nodes = graph.nodes()
train_mask = graph.ndata['train_mask']
val_mask = graph.ndata['val_mask']
test_mask = graph.ndata['test_mask']
label = graph.ndata['label']
feat = graph.ndata['feat']
u = edges[0].numpy()
v = edges[1].numpy()
u_list = edges[0].tolist()
v_list = edges[1].tolist()
partition_data = []
all_edge = 0
for i in range(100):
    (
        g_local, node_feats, edge_feats, gpb, graph_name, ntypes_list, etypes_list,
    ) = dgl.distributed.load_partition(part_config='/root/wtz/RayDGL/dataset/{} {} partition/{}.json'.format(datasetname, num_parts, datasetname), part_id=i)

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
print('all node number: ', sum([partition_data[i][0] for i in range(100)]))
print('all edge number: ', sum([partition_data[i][1] for i in range(100)]))
print('all train number: ', sum([partition_data[i][2] for i in range(100)]))
print('all valid number: ', sum([partition_data[i][3] for i in range(100)]))
print('all test number: ', sum([partition_data[i][4] for i in range(100)]))