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
from ogb.nodeproppred import DglNodePropPredDataset
from degreeJudge import arrangeOutDegree
from degreeNerighbor import findDegreeNerghbor



# 拉取邻接矩阵，特征，标签
# 选择数据集    
argparser = argparse.ArgumentParser(description='决定丢失副本数')
argparser.add_argument('--part_num', 
                       type=int,
                       default=0,
                       help='丢失的子图数')
argparser.add_argument('--loss_time',
                       type=int,
                       default=0,
                       help='选取丢失时间')
argparser.add_argument('--dataset',
                       type=str,
                       default='reddit',
                       help='选取数据集')



argparser.add_argument('--path', type=str, default='/home/asd/文档/wtz/wtz/RayDGL/dataset/igb_small', 
                       help='path containing the datasets')
argparser.add_argument('--dataset_size', type=str, default='small',
                       choices=['tiny', 'small', 'medium', 'large', 'full'], 
                       help='size of the datasets')
argparser.add_argument('--num_classes', type=int, default=19, 
                       choices=[19, 2983], help='number of classes')
argparser.add_argument('--in_memory', type=int, default=1, 
                       choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
argparser.add_argument('--synthetic', type=int, default=0,
                       choices=[0, 1], help='0:nlp-node embeddings, 1:random')
args = argparser.parse_args()


datasetname = args.dataset
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
if datasetname == 'reddit_random':
    dataset = RedditDataset(raw_dir='/home/asd/文档/wtz/wtz/RayDGL/dataset/{}'.format('reddit'))
    num_class = 41
if datasetname == 'flickr':
    dataset = FlickrDataset(raw_dir='/home/asd/文档/wtz/wtz/RayDGL/dataset/{}'.format(datasetname))
    num_class = 7
if datasetname == 'igb_small':
    dataset = IGB260MDGLDataset(args)
    num_class = 19
if datasetname == 'yelp':
    dataset = YelpDataset(raw_dir='/home/asd/文档/wtz/wtz/RayDGL/dataset/{}'.format(datasetname))
    num_class = 100
if datasetname == 'ogbn_products':
    dataset = DglNodePropPredDataset(name = 'ogbn-products', root='/home/asd/文档/wtz/wtz/RayDGL/dataset/ogbn_products')
    num_class = 47
if datasetname == 'ogbn_arxiv':
    dataset = DglNodePropPredDataset(name = 'ogbn-arxiv', root='/home/asd/文档/wtz/wtz/RayDGL/dataset/ogbn_arxiv')
    num_class = 40


if datasetname == 'ogbn_products' or datasetname == 'ogbn_arxiv':
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
    train_mask_origin = graph_full.ndata['train_mask'].bool()
    val_mask_origin = graph_full.ndata['val_mask'].bool()
    test_mask_origin = graph_full.ndata['test_mask'].bool()
    label = graph_full.ndata['label']
    feat = graph_full.ndata['feat']
    u = edges[0].numpy()
    v = edges[1].numpy()
    u_list = edges[0].tolist()
    v_list = edges[1].tolist()
    # print(u)
    # print(v)

def del_edge1(origin_u, origin_v, del_node):
    time = len(origin_u)  # 记录一共所需的循环次数，也就是最开始的列表长度
    j = 0
    for i in range(time):
        if origin_u[j] in del_node or origin_v[j] in del_node:
            origin_u = np.delete(origin_u, j)
            origin_v = np.delete(origin_v, j)
            j-=1
        j+=1
        print('i = {}/{}'.format(i, time))
    return origin_u, origin_v

def del_edge2(origin_u, origin_v, del_node):
    u_v = np.array([origin_u, origin_v])
    j=0
    for i in del_node:  
        position = np.where(u_v == i)    # 找出顶点所在位置
        position_col = position[1].tolist()
        u_v  = np.delete(u_v, position_col, 1)
        j=j+1
        print('j = {}/{}'.format(j, len(del_node)))
    u = u_v[0]
    v = u_v[1]
    return u, v

def del_edge3(origin_u, origin_v, del_node):
    # 创建一个Pandas Series对象
    u_pd = pd.Series(origin_u)
    v_pd = pd.Series(origin_v)
    # 找到所有在u中的del_node的元素的布尔索引
    mask_u = u_pd.isin(del_node)
    # 用布尔索引过滤掉在u中的del_node的元素，同时把v中的也删去
    u_new = u_pd[~mask_u]
    v_new = v_pd[~mask_u]
    # 找到所有在v中的del_node的元素的布尔索引
    mask_v = v_new.isin(del_node)
    # 用布尔索引过滤掉在v中的del_node的元素，同时把u中的也删去
    u_new = np.array(u_new[~mask_v])
    v_new = np.array(v_new[~mask_v])
    return u_new, v_new

def del_edge4(target_node, del_node):    # 从一个里面删另一个里有的
    target = pd.Series(target_node)
    mask_del = target.isin(del_node)
    target_node_new = np.array(target_node[~mask_del])
    return target_node_new
    
def del_mask(del_list, origin_mask):
    n = len(origin_mask)
    for i in del_list:
        origin_mask[i] = False
    return origin_mask

# 选取子图
down = True
# loss_time = 0   # 选择在哪个epoch丢失
loss_time = args.loss_time
rebuild = True
part_num = args.part_num    # 要丢掉的子图数量
# part_num = 0
part_id = random.sample(range(0, num_parts), part_num)       # 随机选取子图
# 手动指定子图处
# part_id = [0]
# part_id = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
# part_id = [25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]
# part_id = [50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74]
part_id = [75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99]   # reddit:98
use_degree_judge = True
ratio = 0.6
hop_range = 0
part_node_id_total = []
for i in range(len(part_id)):
    (
        g_local, node_feats, edge_feats, gpb, graph_name, ntypes_list, etypes_list,
    ) = dgl.distributed.load_partition(part_config='/home/asd/文档/wtz/wtz/RayDGL/dataset/{} {} partition/{}.json'.format(datasetname, num_parts, datasetname), part_id=part_id[i])
    num_node = sum(g_local.ndata['inner_node'].numpy())
    # 生成要删除顶点
    part_node_id = g_local.ndata['_ID'][:num_node]  # 子图中的顶点，也就是要删除的顶点
    
    importantDegreeList = []
    if use_degree_judge == True:
        # 找到当前子图的度排名
        importantDegreeList = arrangeOutDegree(graph=g_local, ratio=ratio)
        # 找到度排名中的顶点的n-hop邻域内顶点
        importantDegreeList = findDegreeNerghbor(graph=g_local, neighborNode=importantDegreeList, hop_range=hop_range)
        # 将这些顶点保留，不删去
        part_node_id = del_edge4(part_node_id, importantDegreeList)  # 子图中的顶点，也就是要删除的顶点
    
    print('partition {}, num_node = {}, important_node = {}'.format(part_id[i], num_node, len(importantDegreeList)))
    part_node_id_total = np.append(part_node_id_total, part_node_id)
    part_node_id_total = part_node_id_total.astype(int)

# 重构mask
new_train_mask = torch.tensor([])
new_val_mask = torch.tensor([])
new_test_mask = torch.tensor([])
for i in range(num_parts):
    (
        g_local, node_feats, edge_feats, gpb, graph_name, ntypes_list, etypes_list,
    ) = dgl.distributed.load_partition(part_config='/home/asd/文档/wtz/wtz/RayDGL/dataset/{} {} partition/{}.json'.format(datasetname, num_parts, datasetname), part_id=i)
    new_train_mask = torch.cat([new_train_mask, node_feats['_N/train_mask']], 0)
    new_val_mask = torch.cat([new_val_mask, node_feats['_N/val_mask']], 0)
    new_test_mask = torch.cat([new_test_mask, node_feats['_N/test_mask']], 0)

train_mask = new_train_mask.bool()
val_mask =new_val_mask.bool()
test_mask =new_test_mask.bool()

# 重构graph
if rebuild == True:
    time_rebuild = time.time()
    print('num_edge = ', len(u))
    new_u, new_v = del_edge3(u, v, part_node_id_total)
    # print(u,len(u),v,len(v))
    # print(new_u,len(new_u),new_v,len(new_v))
    # print(len(part_node_id))
    g = dgl.graph((new_u, new_v), num_nodes=graph_full.num_nodes())
    # train_mask = new_train_mask.bool()  # 转换为bool
    # val_mask = new_val_mask.bool()
    # test_mask = new_test_mask.bool()
    
    # 被删除的顶点不参与训练验证
    train_mask_del = del_mask(part_node_id_total, new_train_mask).bool()  # 删除被删除的子图中的训练顶点,并转换为bool
    val_mask_del = del_mask(part_node_id_total, new_val_mask).bool()
    test_mask_del = del_mask(part_node_id_total, new_test_mask).bool()
    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask
    g.ndata['label'] = label
    g.ndata['feat'] = feat
    time_rebuild = time.time() - time_rebuild
    print('time_rebuild = ', time_rebuild)
else:
    g = graph_full

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_size, num_classes, allow_zero_in_degree=True)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h

class SAGE(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super().__init__()
        self.conv1 = SAGEConv( 
            in_feats=in_feats, out_feats=hidden_size, aggregator_type='gcn')
        self.conv2 = SAGEConv(
            in_feats=hidden_size, out_feats=40, aggregator_type='mean')
        self.conv3 = SAGEConv(
            in_feats=40, out_feats=num_classes, aggregator_type='mean')
        # self.dropout =  nn.Dropout(dropout)
    
    def forward(self, graph, inputs):
        # inputs 是节点的特征 [N, in_feas]
        h = self.conv1(graph, inputs)
        h = torch.relu(h)
        h = self.conv2(graph, h)
        h = torch.relu(h)
        h = self.conv3(graph, h)
        return h 

def evaluate1(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def evaluate2(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        softmax = torch.nn.Softmax(dim=1)
        logits = softmax(logits)
        logits = logits > 0.01  # 假设使用阈值0.5进行多标签分类
        # for i in range(labels.shape[0]):
        #     if sum(labels[i]) == 0:
        #         continue
        median1 = np.logical_and(labels.cpu(), logits.cpu())
        median1 = np.sum(median1.cpu().numpy(), axis=1)
        median2 = np.sum(labels.cpu().numpy(), axis=1)
        nonzero_rows = median2 != 0     # 找出全0行索引
        median2 = median2[nonzero_rows]     # 删除全0行
        median1 = median1[nonzero_rows]     # 把median2全零行在median1中也删掉
        count = median1 / median2
        return np.sum(count) / count.shape[0]

model = SAGE(feat.shape[1], 128, num_classes=num_class)
if datasetname == 'yelp':
    loss_function = nn.MultiLabelSoftMarginLoss()
else:
    loss_function = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters())

# GPU上训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
loss_function.to(device)
feat = feat.to(device)
graph_full = graph_full.to(device)
g = g.to(device)
label = label.to(device)

time1 = time.time()
losses = []
best_val_acc = 0
for epoch in range(300):
    print('Epoch {}'.format(epoch))
    model.train()
    
    graph = graph_full
    if down == True:
        if epoch >= loss_time:
            graph = g
            train_mask = train_mask_del
            val_mask = val_mask_del
            # test_mask = test_mask_del
    
    # graph = g
    
    # 用所有的节点进行前向传播
    logits = model(graph, feat)
    # 计算损失
    loss = loss_function(logits[train_mask], label[train_mask])
    # 计算验证集accuracy
    if datasetname == 'yelp':
        acc = evaluate2(model, g, feat, label, val_mask)
    else:
        acc = evaluate1(model, g, feat, label, val_mask)
    # backward propagation
    opt.zero_grad()
    loss.backward()
    opt.step()
    print('loss = {:.4f},  acc = {:.4f}'.format(loss.item(), acc))
    losses.append(loss.detach().cpu().numpy())
print("\nTotal Time = ", time.time() - time1)

# 测试集评估
model.eval()
if datasetname == 'yelp':
    acc_test = evaluate2(model, graph_full, feat, label, test_mask)
else:
    acc_test = evaluate1(model, graph_full, feat, label, test_mask)

print("test accuracy = {:.4f}".format(acc_test))

if down == True:
    print('part_id: ', part_id)
print('train_node: ', sum(train_mask.tolist()))
print('valid_node: ', sum(val_mask.tolist()))
print('test_node: ', sum(test_mask.tolist()))

