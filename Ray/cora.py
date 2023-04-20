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


class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, dropout=0.2):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, feat_drop=0.2, aggregator_type='gcn')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, feat_drop=0.2, aggregator_type='mean')
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph, inputs):
        # inputs 是节点的特征 [N, in_feas]
        h = self.conv1(graph, inputs)
        h = self.dropout(F.relu(h))
        h = self.conv2(graph, h)
        return h


dataset = CoraGraphDataset() # Cora citation network dataset
graph = dataset[0]
graph = dgl.remove_self_loop(graph)  # 消除自环
node_features = graph.ndata['feat']
node_labels = graph.ndata['label']
train_mask = graph.ndata['train_mask']
valid_mask = graph.ndata['val_mask']
test_mask = graph.ndata['test_mask']
n_features = node_features.shape[1]
n_labels = int(node_labels.max().item() + 1)

print("图的节点数和边数: ", graph.num_nodes(), graph.num_edges())
print("训练集节点数：", train_mask.sum().item())
print("验证集集节点数：", valid_mask.sum().item())
print("测试集节点数：", test_mask.sum().item())
print("节点特征维数：", n_features)
print("标签类目数：", n_labels)

G = graph.to_networkx()
res = np.random.randint(0, high=G.number_of_nodes(), size=(200))

k = G.subgraph(res)
pos = nx.kamada_kawai_layout(k)
# pos = nx.spring_layout(k)

# 随机选取节点展示
plt.figure()
nx.draw(k, pos=pos, node_size=8 )
# plt.savefig('cora.jpg', dpi=600)
plt.show()


def evaluate(model, graph, input_features, output_labels, mask):
    model.eval()
    with torch.no_grad():
        output_features = model(graph, input_features)
        output_features = output_features[mask]
        output_labels = output_labels[mask]
        _, indices = torch.max(output_features, dim=1)
        correct = torch.sum(indices == output_labels)
        return correct.item() * 1.0 / len(output_labels)


model = SAGE(in_feats=n_features, hid_feats=128, out_feats=n_labels)
loss_function = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters())

# GPU上训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
loss_function.to(device)
node_features = node_features.to(device)
graph = graph.to(device)
node_labels = node_labels.to(device)


# 开始训练
time1 = time.time()
losses = []
best_val_acc = 0
for epoch in range(200):
    print('Epoch {}'.format(epoch))
    model.train()
    # 用所有的节点进行前向传播
    output = model(graph, node_features)
    # 计算损失
    loss = loss_function(output[train_mask], node_labels[train_mask])
    # 计算验证集accuracy
    acc = evaluate(model, graph, node_features, node_labels, valid_mask)
    # backward propagation
    opt.zero_grad()
    loss.backward()
    opt.step()
    print('loss = {:.4f}'.format(loss.item()))
    if acc > best_val_acc:
        best_val_acc = acc
        torch.save(model.state_dict(), "static/best_model.pth")
    print("current val acc = {}, best val acc = {}".format(acc, best_val_acc))
    losses.append(loss.detach().cpu().numpy())
print("Total Time:", time.time() - time1)

# 测试集评估
model.load_state_dict(torch.load("static/best_model.pth"))
acc = evaluate(model, graph, node_features, node_labels, test_mask)
print("test accuracy: ", acc)


plt.plot(losses)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()