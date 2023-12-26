import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import time
import argparse
from dgl.data import *
import tqdm
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')
argparser = argparse.ArgumentParser("edge classification training")
argparser.add_argument('--num-epochs', type=int, default=30)
argparser.add_argument('--num-hidden', type=int, default=64)
argparser.add_argument('--num-layers', type=int, default=2)
argparser.add_argument('--fan-out', type=str, default='10,25')  ## 第一层随机采样10个，第二层采样25个
argparser.add_argument('--batch-size', type=int, default=2048)
argparser.add_argument('--lr', type=float, default=0.005)
argparser.add_argument('--dropout', type=float, default=0.5)
argparser.add_argument('--num-workers', type=int, default=0,
                       help="Number of sampling processes. Use 0 for no extra process.")
args = argparser.parse_args([])

# load wordnet data
data = WN18Dataset(raw_dir='/home/asd/文档/wtz/wtz/RayDGL/dataset/wn18')
n_classes = data.num_rels   ## 关系数量，也就是边分类的分类数
g = data[0]
## 训练集、验证集、测试集
train_edge_mask = g.edata.pop('train_edge_mask')
val_edge_mask = g.edata.pop('valid_edge_mask')
test_edge_mask = g.edata.pop('test_edge_mask')

# Pack data
data = train_edge_mask, val_edge_mask, test_edge_mask, n_classes, g
print('\n', g)
n_edges = g.num_edges()   # 图中边的数量
labels = g.edata['etype']  # 图中所有边的标签


train_edge_mask, val_edge_mask, test_edge_mask, n_classes, g = data
print('\n', train_edge_mask.sum(), val_edge_mask.sum(), test_edge_mask.sum())

## train, valid, test 边的id列表
train_eid = th.LongTensor(np.nonzero(train_edge_mask)).squeeze()
val_eid = th.LongTensor(np.nonzero(val_edge_mask)).squeeze()
test_eid = th.LongTensor(np.nonzero(test_edge_mask)).squeeze()
print(train_eid.shape, val_eid.shape, test_eid.shape)

# Create sampler
sampler = dgl.dataloading.MultiLayerNeighborSampler(
            [int(fanout) for fanout in args.fan_out.split(',')])
dataloader = dgl.dataloading.DataLoader(
        g, train_eid, sampler,
        # exclude='reverse_id',    # 去除反向边，否则模型可能知道存在边的联系，导致模型“作弊”
        # For each edge with ID e in dataset, the reverse edge is e ± |E|/2.
        # reverse_eids=th.cat([th.arange(n_edges // 2, n_edges), th.arange(0, n_edges // 2)]),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

## For debug
for dd in dataloader:
    """分别是input_nodes, edge_subgraph，blocks"""
    for xx in dd:
        print(xx)
        print()
    break

class TwoLayerGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        """两层的GCN模型"""
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hid_dim, allow_zero_in_degree=True)
        self.conv2 = dglnn.GraphConv(hid_dim, out_dim, allow_zero_in_degree=True)
    
    def forward(self, blocks, x):
        x = F.relu(self.conv1(blocks[0], x))
        x = F.relu(self.conv2(blocks[1], x))
        return x
    
class Predictor(nn.Module):
    """边预测模块，将边两端节点表示拼接，然后接一个线性变换，得到最后的分类表示输出"""
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.W = nn.Linear(2 * in_dim, num_classes)
        
    def apply_edges(self, edges):
        data = th.cat([edges.src['x'], edges.dst['x']], dim=-1)
        return {'score': self.W(data)}
    
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            edge_subgraph.apply_edges(self.apply_edges)
            return edge_subgraph.edata['score']

class MyModel(nn.Module):
    """主模型：结构比较清晰"""
    def __init__(self, emb_dim, hid_dim, out_dim, num_classes, num_nodes):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, emb_dim)  
        self.gcn = TwoLayerGCN(emb_dim, hid_dim, out_dim)
        self.predictor = Predictor(out_dim, num_classes)
        
    def forward(self, edge_subgraph, blocks, input_nodes):
        x = self.node_emb(input_nodes)
        x = self.gcn(blocks, x)
        return self.predictor(edge_subgraph, x)
    
    
device = th.device("cuda" if th.cuda.is_available() else "cpu")
print("device: {}".format(device))

model = MyModel(64, args.num_hidden, args.num_hidden, 18, g.num_nodes())
model = model.to(device)
loss_fcn = nn.CrossEntropyLoss()   # 交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def predict(model, g, valid_eid, device):
    # Create sampler（全采样）
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.EdgeDataLoader(
        g, valid_eid, sampler, exclude='reverse_id',
        # For each edge with ID e in dataset, the reverse edge is e ± |E|/2.
        reverse_eids=th.cat([th.arange(n_edges // 2, n_edges), th.arange(0, n_edges // 2)]),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers)
    
    valid_preds = []
    model.eval()
    with th.no_grad():
        for input_nodes, edges_subgraph, blocks in dataloader:
            edges_subgraph = edges_subgraph.to(device)
            blocks = [block.int().to(device) for block in blocks]
            pred = model(edges_subgraph, blocks, input_nodes)
            pred = pred.cpu().argmax(-1).numpy().tolist()
            valid_preds.extend(pred)
    return valid_preds

best_val_acc = 0  # 记录验证集上的最好效果
patience = 0  # For early stopping

# Training loop
for epoch in range(args.num_epochs):
    # Loop over the dataloader to sample the computation dependency graph as a list of
    # blocks.
    start_time = time.time()
    all_loss = []
    trn_label, trn_pred = [], []
    n_batches = len(dataloader)
    
    for step, (input_nodes, edges_subgraph, blocks) in enumerate(dataloader):
        edges_subgraph = edges_subgraph.to(device)
        blocks = [block.to(device) for block in blocks]

        # Compute loss and prediction
        edge_preds = model(edges_subgraph, blocks, input_nodes)
        loss = loss_fcn(edge_preds, edges_subgraph.edata['etype'])  # or labels[edges_subgraph.edata['_ID']]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        all_loss.append(loss.item())
        trn_label.extend(edges_subgraph.edata['etype'].cpu().numpy().tolist())
        trn_pred.extend(edge_preds.argmax(-1).detach().cpu().numpy().tolist())
        
        if (step+1) % (n_batches//10) == 0:
            cur_acc = metrics.accuracy_score(trn_label, trn_pred)
            print('Epoch {:2d} | Step {:04d}/{} | Loss {:.4f} | Avg Loss {:.4f} | Acc {:.4f} | Time {:.2f}s'.format(
                  epoch+1, step, n_batches, loss.item(), np.mean(all_loss), cur_acc, time.time() - start_time))

    ## 验证集预测
    val_preds = predict(model, g, val_eid, device)
    val_acc = metrics.accuracy_score(labels[val_eid], val_preds)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience = 0
        th.save(model.state_dict(), "edge_cls_best_model.bin")
    else:
        patience += 1
    print('Cur Val Acc {:.4f}, Best Val Acc {:.4f}, Time {:.2f}s'.format(
          val_acc, best_val_acc, time.time() - start_time))
    
    ## earlystopping，如果验证集效果连续三次以上都没上升，直接停止训练
    if patience > 3:
        break
    
model.load_state_dict(th.load("edge_cls_best_model.bin"))

test_preds = predict(model, g, test_eid, device)
print("test acc: {:.4f}".format(metrics.accuracy_score(labels[test_eid], test_preds)))