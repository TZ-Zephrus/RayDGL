import random
import pandas as pd
import numpy as np
import numpy.random
import torch
import dgl
import torch.nn as nn
from dgl.data import CoraGraphDataset, PubmedGraphDataset, RedditDataset
from dgl.nn.pytorch import GraphConv, SAGEConv
import time
import networkx as nx

# 拉取邻接矩阵，特征，标签
# 选择数据集
datasetname = 'pubmed'


if datasetname == 'cora':
    dataset = CoraGraphDataset(raw_dir='/home/asd/文档/wtz/wtz/RayDGL/dataset/{}'.format(datasetname))
    num_class = 7
if datasetname == 'cora_nobalance':
    dataset = CoraGraphDataset(raw_dir='/home/asd/文档/wtz/wtz/RayDGL/dataset/{}'.format(datasetname))
    num_class = 7
if datasetname == 'pubmed':
    dataset = PubmedGraphDataset(raw_dir='/home/asd/文档/wtz/wtz/RayDGL/dataset/{}'.format(datasetname))
    num_class = 3
if datasetname == 'pubmed_random':
    dataset = PubmedGraphDataset(raw_dir='/home/asd/文档/wtz/wtz/RayDGL/dataset//{}'.format('pubmed'))
    num_class = 3
if datasetname == 'reddit':
    dataset = RedditDataset(raw_dir='/home/asd/文档/wtz/wtz/RayDGL/dataset/{}'.format(datasetname))
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
# np.save('D:/1a一些学习/一些代码/graph NN/dataset/data_u.npy', u)
# np.save('D:/1a一些学习/一些代码/graph NN/dataset/data_v.npy', v)
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
    # 波浪线在pandas中代表否定
    u_new = u_pd[~mask_u]
    v_new = v_pd[~mask_u]
    # 找到所有在v中的del_node的元素的布尔索引
    mask_v = v_new.isin(del_node)
    # 用布尔索引过滤掉在v中的del_node的元素，同时把u中的也删去
    u_new = np.array(u_new[~mask_v])
    v_new = np.array(v_new[~mask_v])
    return u_new, v_new

def del_mask(del_list, origin_mask):
    n = len(origin_mask)
    for i in del_list:
        origin_mask[i] = False
    return origin_mask

# 选取子图
down = False
loss_time = 0   # 选择在哪个epoch丢失
rebuild = False
part_num = 2    # 要丢掉的子图数量
part_id = random.sample(range(0, 100), part_num)       # 随机选取子图
# 手动指定子图处
# part_id = [0]

part_node_id_total = []
for i in range(len(part_id)):
    (
        g_local, node_feats, edge_feats, gpb, graph_name, ntypes_list, etypes_list,
    ) = dgl.distributed.load_partition(part_config='/home/asd/文档/wtz/wtz/RayDGL/dataset/{} 100 partition/{}.json'.format(datasetname, datasetname), part_id=part_id[i])

    num_node = sum(g_local.ndata['inner_node'].numpy())
    print('partition {}, num_node = {}'.format(part_id[i], num_node))
    part_node_id = g_local.ndata['_ID'][:num_node]  # 子图中的顶点，也就是要删除的顶点
    part_node_id_total = np.append(part_node_id_total, part_node_id)
    part_node_id_total = part_node_id_total.astype(int)

# np.save('D:/1a一些学习/一些代码/graph NN/dataset/del_node.npy', part_node_id_total)

# 重构mask
new_train_mask = torch.tensor([])
new_val_mask = torch.tensor([])
new_test_mask = torch.tensor([])
for i in range(100):
    (
        g_local, node_feats, edge_feats, gpb, graph_name, ntypes_list, etypes_list,
    ) = dgl.distributed.load_partition(part_config='/home/asd/文档/wtz/wtz/RayDGL/dataset/{} 100 partition/{}.json'.format(datasetname, datasetname), part_id=i)
    new_train_mask = torch.cat([new_train_mask, node_feats['_N/train_mask']], 0)
    new_val_mask = torch.cat([new_val_mask, node_feats['_N/val_mask']], 0)
    new_test_mask = torch.cat([new_test_mask, node_feats['_N/test_mask']], 0)

train_mask = new_train_mask.bool()
val_mask = new_val_mask.bool()
test_mask = new_test_mask.bool()

# 重构graph
if rebuild == True:
    time_rebuild = time.time()
    print('num_edge = ', len(u))
    new_u, new_v = del_edge3(u, v, part_node_id_total)
    # print(u,len(u),v,len(v))
    # print(new_u,len(new_u),new_v,len(new_v))
    # print(len(part_node_id))
    g = dgl.graph((new_u, new_v), num_nodes=graph.num_nodes())
    # train_mask = new_train_mask.bool()  # 转换为bool
    # val_mask = new_val_mask.bool()
    # test_mask = new_test_mask.bool()
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
    g = graph

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

def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

model = GCN(feat.shape[1], 128, num_classes=num_class)
loss_function = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters())

# GPU上训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
loss_function.to(device)
feat = feat.to(device)
graph = graph.to(device)
g = g.to(device)
label = label.to(device)

time1 = time.time()
losses = []
best_val_acc = 0
for epoch in range(500):
    print('Epoch {}'.format(epoch))
    model.train()
    graph = graph
    if down == True:
        # 当epoch数大于丢失顶点的epoch时，使用不完整的图和删除过的mask进行训练
        if epoch >= loss_time:
            graph = g
            train_mask = train_mask_del
            val_mask = val_mask_del
            test_mask = test_mask_del
    # 用所有的节点进行前向传播
    logits = model(graph, feat)
    # 计算损失
    loss = loss_function(logits[train_mask], label[train_mask])
    # 计算验证集accuracy
    acc = evaluate(model, graph, feat, label, val_mask)
    # backward propagation
    opt.zero_grad()
    loss.backward()
    opt.step()
    print('loss = {:.4f},  acc = {:.4f}'.format(loss.item(), acc))
    losses.append(loss.detach().cpu().numpy())
print("\nTotal Time = ", time.time() - time1)

# 测试集评估
acc = evaluate(model, graph, feat, label, test_mask)
print("test accuracy = {:.4f}".format(acc))

if down == True:
    print('part_id: ', part_id)
print('train_node: ', sum(train_mask.tolist()))
print('valid_node: ', sum(val_mask.tolist()))
print('test_node: ', sum(test_mask.tolist()))







