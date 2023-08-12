import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import dgl
import dgl.nn as dglnn

# 分布式训练函数
def train(rank, world_size):
    # 创建图数据
    graph = dgl.graph(([0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 0]))

    # 创建模型
    class GNN(nn.Module):
        def __init__(self):
            super(GNN, self).__init__()
            self.conv1 = dglnn.GraphConv(1, 16)
            self.conv2 = dglnn.GraphConv(16, 1)
        
        def forward(self, g, features):
            x = torch.sin(features)  # 假设使用 sin 函数作为特征变换
            x = self.conv1(g, x)
            x = torch.relu(x)
            x = self.conv2(g, x)
            return x

    model = GNN().to(rank)

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # 分布式数据并行
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # 设置模型为训练模式
    model.train()

    # 将数据移动到对应的设备上
    features = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7]], dtype=torch.float32).to(rank)
    labels = torch.tensor([[0.2], [0.4], [0.6], [0.8], [1.0], [1.2], [1.4]], dtype=torch.float32).to(rank)
    graph = graph.to(rank)

    # 执行训练过程
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(graph, features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        print(f'Rank {rank}, Epoch [{epoch+1}/100], Loss: {loss.item()}')

# 启动分布式训练
def run_training(rank, world_size):
    # 初始化分布式训练
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:12345', rank=rank, world_size=world_size)

    # 执行分布式训练
    train(rank, world_size)

    # 清理分布式训练环境
    dist.destroy_process_group()

# 启动多进程分布式训练
if __name__ == '__main__':
    world_size = 2
    mp.spawn(run_training, args=(world_size, ), nprocs=world_size)
