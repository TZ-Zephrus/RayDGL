import dgl.nn as dglnn
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import CoraGraphDataset
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
import torchmetrics.functional as MF
import ray

class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, dropout=0.2):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, feat_drop=0.2, aggregator_type='gcn')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, feat_drop=0.2, aggregator_type='mean')
        self.dropout = nn.Dropout(dropout)

    def forward(self, block, inputs):
        # inputs 是节点的特征 [N, in_feas]
        h = self.conv1(block[0], inputs)
        h = self.dropout(F.relu(h))
        h = self.conv2(block[-1], h)
        return h

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


# 数据准备
sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2) # 这里采样得到两个块
dataset = CoraGraphDataset(raw_dir="/root/wtz/dataset/cora") # Cora citation network dataset
graph = dataset[0]
graph = dgl.remove_self_loop(graph)  # 消除自环
node_features = graph.ndata['feat']
node_labels = graph.ndata['label']
n_features = node_features.shape[1]
n_labels = int(node_labels.max().item() + 1)

# 准备ray类内数据
feats = graph.ndata['feat']
train_mask = graph.ndata['train_mask']
valid_mask = graph.ndata['val_mask']
test_mask = graph.ndata['test_mask']
labels = graph.ndata['label']
g = []
train_idx = []
train_dataloader = []
for i in range(3):
    (
    g_local, node_feats, edge_feats, gpb, graph_name, ntypes_list, etypes_list,
    ) = dgl.distributed.load_partition('/root/wtz/dataset/cora partition/cora.json', i)
    
    local_nodes = g_local.ndata['_ID']
    inner_nodes_id = torch.nonzero(g_local.ndata['inner_node']==1, as_tuple=False).view(-1)
    outer_nodes_id = torch.nonzero(g_local.ndata['inner_node']==0, as_tuple=False).view(-1)
    inner_nodes = local_nodes[inner_nodes_id] # inner_nodes: global IDs of inner nodes
    outer_nodes = local_nodes[outer_nodes_id] # outer_nodes: global IDs of outer nodes

    local_feats = torch.cat([feats[inner_nodes], feats[outer_nodes]], dim=0)
    # 这里可能有点小问题 因为我是把node_feats中的903个特征直接按上去，假设了子图g_local中的前903个点和node_feats中的903个特征是一一对应的
    while len(node_feats['_N/train_mask']) != g_local.num_nodes():
        false_column = torch.zeros((1,), dtype=torch.bool)
        node_feats['_N/train_mask'] = torch.cat((node_feats['_N/train_mask'], false_column), dim=0)
        node_feats['_N/val_mask'] = torch.cat((node_feats['_N/val_mask'], false_column), dim=0)
        node_feats['_N/test_mask'] = torch.cat((node_feats['_N/test_mask'], false_column), dim=0)
    local_labels = torch.cat([labels[inner_nodes], labels[outer_nodes]], dim=0)
    g_local.ndata['feat'] = local_feats
    g_local.ndata['train_mask'] = node_feats['_N/train_mask']
    g_local.ndata['val_mask'] = node_feats['_N/val_mask']
    g_local.ndata['test_mask'] = node_feats['_N/test_mask']
    g_local.ndata['label'] = local_labels
    print(f"partition {i} has {local_nodes.size()} total nodes, "
        f"where {inner_nodes.size()} innder nodes and {outer_nodes.size()} outer nodes")
    # 创建一个三个进程的子图列表
    g.append(g_local)
    
    
    # 创建一个包含三个进程上的子图的train_idx的列表 train_idx=[[1,2,...], [903, 904...], []]
    train_idxx = []
    for j in range(g[i].num_nodes()):
        if g[i].ndata['train_mask'][j] == True:
            train_idxx.append(j)
    train_idx.append(train_idxx)
    # 创建一个三个进程的train_dataloader列表
    train_dataloader_local = dgl.dataloading.DataLoader(
        g[i], 
        train_idx[i],  
        sampler,
        batch_size=512,
        shuffle=True,
        drop_last=False,
        num_workers=4)
    # for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader_local):
    #     continue
    train_dataloader.append(train_dataloader_local)

# 验证集和测试集不需要分布式
# 建立验证 测试 集索引
valid_idx = []
test_idx = []
for i in range(graph.num_nodes()):
    if valid_mask[i] == True:
        valid_idx.append(i)
    if test_mask[i] == True:
        test_idx.append(i)
        
valid_dataloader = dgl.dataloading.DataLoader(
    graph, 
    valid_idx,  
    sampler,
    batch_size=1024,
    shuffle=True,
    drop_last=False,
    num_workers=4)

test_dataloader = dgl.dataloading.DataLoader(
    graph, 
    test_idx,  
    sampler,
    batch_size=1024,
    shuffle=True,
    drop_last=False,
    num_workers=4)


def evaluate(model, dataloader):
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        model.eval()
        input_features = []
        output_labels = []
        output_features = []
        with torch.no_grad():
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[1].dstdata['label']
            output_features = model(blocks, input_features)
            # return MF.accuracy(output_features, output_labels, task="multiclass", num_classes=7)
            output_labels = output_labels.numpy()
            output_features = torch.max(output_features, dim=1)[1].numpy()
            acc = (output_features == output_labels).sum()/len(output_labels)
    return acc


loss_function = nn.CrossEntropyLoss()

# 选择device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# self.model.to(device)
# loss_function.to(device)


@ray.remote
class ParameterServer(object):
    def __init__(self):
        self.model = SAGE(in_feats=n_features, hid_feats=128, out_feats=n_labels)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def apply_gradients(self, *gradients):
        summed_gradients = [np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*gradients)]
        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        return self.model.get_weights()

    def get_weights(self):
        return self.model.get_weights()


@ray.remote
class DataWorker(object):
    def __init__(self):
        self.model = SAGE(in_feats=n_features, hid_feats=128, out_feats=n_labels)
        self.g = iter(g)
        self.train_dataloader = iter(train_dataloader)

    def compute_gradients(self, weights):
        self.model.set_weights(weights)
        try:
            # local_graph = next(self.g)
            local_train_dataloader = next(self.train_dataloader)# 本进程内的train_dataloader
        except StopIteration:  # When the epoch ends, start a new epoch.
            self.g = iter(g)
            self.train_dataloader = iter(train_dataloader)
            # local_graph = next(self.g)
            local_train_dataloader = next(self.train_dataloader)
        for it, (input_nodes, output_nodes, blocks) in enumerate(local_train_dataloader):
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label']
            output_features = self.model(blocks, input_features)
            # 计算损失
            loss = loss_function(output_features, output_labels)
            # backward propagation
            self.model.zero_grad()
            loss.backward()
        return self.model.get_gradients()


epoch = 200
num_workers = 3

ray.init(ignore_reinit_error=True)
ps = ParameterServer.remote()
workers = [DataWorker.remote() for i in range(num_workers)]

# 在驱动过程中实例化一个模型，以评估训练期间的测试准确性。
model = SAGE(in_feats=n_features, hid_feats=128, out_feats=n_labels)


'''
训练交替进行：
1、计算给定服务器当前权重的梯度
2、使用梯度更新参数服务器的权重。
'''
time1 = time.time()
print("Running synchronous parameter server training.\ntraining...")
current_weights = ps.get_weights.remote()
for i in range(epoch):
    gradients = [worker.compute_gradients.remote(current_weights) for worker in workers]
    # Calculate update after all gradients are available.
    current_weights = ps.apply_gradients.remote(*gradients)
    # Evaluate the current model.
    model.set_weights(ray.get(current_weights))
    accuracy = evaluate(model, valid_dataloader)
    print("Epoch: {}  accuracy: {:.4f}".format(i, accuracy))


# 测试集评估
print("\ntesting...")
acc = evaluate(model, test_dataloader)
print("test accuracy:  {:.4f}".format(acc))
print("time: {:.4f}".format(time.time() - time1))
# Clean up Ray resources and processes before the next example.
ray.shutdown()