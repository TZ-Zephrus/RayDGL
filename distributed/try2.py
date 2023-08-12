import torch
import torch.nn as nn
import torch.optim as optim
import dgl
import dgl.function as fn
from dgl.nn import GraphConv

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建图数据
graph = dgl.graph(([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]))

# 定义 GNN 模型
class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GraphConv(1, 16)  # 输入特征维度为 1，输出特征维度为 16
        self.conv2 = GraphConv(16, 1)  # 输入特征维度为 16，输出特征维度为 1

    def forward(self, g, features):
        x = torch.sin(features)  # 假设使用 sin 函数作为特征变换
        x = self.conv1(g, x)
        x = torch.relu(x)
        x = self.conv2(g, x)
        return x

# 创建 GNN 模型实例
model = GNN()

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 分布式训练
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=[0, 1])  # 指定使用的 GPU 设备
model.train()  # 设置模型为训练模式

# 设置模型为训练模式
model.train()

# 将数据和模型移动到对应的设备上
features = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7]], dtype=torch.float32).to(device)
labels = torch.tensor([[0.2], [0.4], [0.6], [0.8], [1.0], [1.2], [1.4]], dtype=torch.float32).to(device)
graph = graph.to(device)

# 执行训练过程
for epoch in range(100):
    optimizer.zero_grad()
    features = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]], dtype=torch.float32)
    labels = torch.tensor([[0.2], [0.4], [0.6], [0.8], [1.0], [1.2]], dtype=torch.float32)
    outputs = model(graph, features)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 100, loss.item()))
