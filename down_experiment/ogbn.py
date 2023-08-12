import argparse
import time

import dgl
import dgl.nn.pytorch as dglnn
import pandas as pd
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from ogb.nodeproppred import DglNodePropPredDataset
import random

class SAGE(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[: block.num_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for l, layer in enumerate(self.layers):
            y = th.zeros(
                g.num_nodes(),
                self.n_hidden if l != len(self.layers) - 1 else self.n_classes,
            ).to(device)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                g,
                th.arange(g.num_nodes()),
                sampler,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=args.num_workers,
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].int().to(device)

                h = x[input_nodes]
                h_dst = h[: block.num_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h

            x = y
        return y


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, nfeat, labels, val_nid, test_nid, device):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, nfeat, device)
    model.train()
    return (
        compute_acc(pred[val_nid], labels[val_nid]),
        compute_acc(pred[test_nid], labels[test_nid]),
        pred,
    )


def load_subtensor(nfeat, labels, seeds, input_nodes):
    """
    Extracts features and labels for a set of nodes.
    """
    batch_inputs = nfeat[input_nodes]
    batch_labels = labels[seeds]
    return batch_inputs, batch_labels


#### Entry point
def run(args, device, data):
    # Unpack data
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, nfeat, g, train_nid_full, val_nid_full, test_nid_full, graph_full = data

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")]
    )
    dataloader = dgl.dataloading.DataLoader(
        g,
        train_nid,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    # Define model and optimizer
    model = SAGE(
        in_feats,
        args.num_hidden,
        n_classes,
        args.num_layers,
        F.relu,
        args.dropout,
    )
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Training loop
    avg = 0
    iter_tput = []
    best_eval_acc = 0
    best_test_acc = 0
    for epoch in range(args.num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            tic_step = time.time()

            # copy block to gpu
            blocks = [blk.int().to(device) for blk in blocks]

            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(
                nfeat, labels, seeds, input_nodes
            )

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = (
                    th.cuda.max_memory_allocated() / 1000000
                    if th.cuda.is_available()
                    else 0
                )
                print(
                    "Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MB".format(
                        epoch,
                        step,
                        loss.item(),
                        acc.item(),
                        np.mean(iter_tput[3:]),
                        gpu_mem_alloc,
                    )
                )

        toc = time.time()
        print("Epoch Time(s): {:.4f}".format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch != 0:
            eval_acc, test_acc, pred = evaluate(
                model, graph_full, nfeat, labels, val_nid, test_nid_full, device
            )
            if args.save_pred:
                np.savetxt(
                    args.save_pred + "%02d" % epoch,
                    pred.argmax(1).cpu().numpy(),
                    "%d",
                )
            print("Eval Acc {:.4f}".format(eval_acc))
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                best_test_acc = test_acc
            print(
                "Best Eval Acc {:.4f} Test Acc {:.4f}".format(
                    best_eval_acc, best_test_acc
                )
            )

    print("Avg epoch time: {}".format(avg / (epoch - 4)))
    return best_test_acc


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID. Use -1 for CPU training",
    )
    argparser.add_argument("--num-epochs", type=int, default=20)
    argparser.add_argument("--num-hidden", type=int, default=256)
    argparser.add_argument("--num-layers", type=int, default=3)
    argparser.add_argument("--fan-out", type=str, default="5,10,15")
    argparser.add_argument("--batch-size", type=int, default=1000)
    argparser.add_argument("--val-batch-size", type=int, default=10000)
    argparser.add_argument("--log-every", type=int, default=20)
    argparser.add_argument("--eval-every", type=int, default=1)
    argparser.add_argument("--lr", type=float, default=0.003)
    argparser.add_argument("--dropout", type=float, default=0.5)
    argparser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of sampling processes. Use 0 for no extra process.",
    )
    argparser.add_argument("--save-pred", type=str, default="")
    argparser.add_argument("--wd", type=float, default=0)
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device("cuda:%d" % args.gpu)
    else:
        device = th.device("cpu")

    
    
    # 多次运行
    def multi_run(p):
        # load ogbn-products data
        dataset = DglNodePropPredDataset(name="ogbn-products", root='/home/asd/文档/wtz/wtz/RayDGL/dataset/ogbn_products')
        
        
        datasetname = 'ogbn_products'
        num_parts = 100
        graph_full, label = dataset[0] # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
        edges = graph_full.edges()
        nodes = graph_full.nodes()
        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        train_mask = th.zeros(graph_full.num_nodes(), dtype=bool)
        val_mask = th.zeros(graph_full.num_nodes(), dtype=bool)
        test_mask = th.zeros(graph_full.num_nodes(), dtype=bool)
        train_mask[train_idx] = True
        val_mask[valid_idx] = True
        test_mask[test_idx] = True
        label = label.flatten()           # 张量展开为1维
        feat = graph_full.ndata['feat']    
        u = edges[0].numpy()
        v = edges[1].numpy()
        u_list = edges[0].tolist()
        v_list = edges[1].tolist()
        
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
        
        def del_mask(del_list, origin_mask):
            n = len(origin_mask)
            for i in del_list:
                origin_mask[i] = False
            return origin_mask
        
        # 选取子图
        down = True
        # loss_time = 0   # 选择在哪个epoch丢失
        loss_time = 0
        rebuild = True
        part_num = p    # 要丢掉的子图数量
        # part_num = 0
        part_id = random.sample(range(0, num_parts), part_num)       # 随机选取子图
        # 手动指定子图处
        # part_id = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
        # part_id = [25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]
        # part_id = [50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74]
        # part_id = [75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99]
        
        delete = True
        
        part_node_id_total = []
        for i in range(len(part_id)):
            (
                g_local, node_feats, edge_feats, gpb, graph_name, ntypes_list, etypes_list,
            ) = dgl.distributed.load_partition(part_config='/home/asd/文档/wtz/wtz/RayDGL/dataset/{} {} partition/{}.json'.format(datasetname, num_parts, datasetname), part_id=part_id[i])

            num_node = sum(g_local.ndata['inner_node'].numpy())
            print('partition {}, num_node = {}'.format(part_id[i], num_node))
            part_node_id = g_local.ndata['_ID'][:num_node]  # 子图中的顶点，也就是要删除的顶点
            part_node_id_total = np.append(part_node_id_total, part_node_id)
            part_node_id_total = part_node_id_total.astype(int)

        # 重构mask
        new_train_mask = th.tensor([])
        new_val_mask = th.tensor([])
        new_test_mask = th.tensor([])
        for i in range(num_parts):
            (
                g_local, node_feats, edge_feats, gpb, graph_name, ntypes_list, etypes_list,
            ) = dgl.distributed.load_partition(part_config='/home/asd/文档/wtz/wtz/RayDGL/dataset/{} {} partition/{}.json'.format(datasetname, num_parts, datasetname), part_id=i)
            new_train_mask = th.cat([new_train_mask, node_feats['_N/train_mask']], 0)
            new_val_mask = th.cat([new_val_mask, node_feats['_N/val_mask']], 0)
            new_test_mask = th.cat([new_test_mask, node_feats['_N/test_mask']], 0)

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
        
        
        train_idx_full = np.where(train_mask)[0]
        val_idx_full = np.where(val_mask)[0]
        test_idx_full = np.where(test_mask)[0]
        
        if delete == True:
            # 生成不完整图的索引
            train_mask = train_mask_del
            val_mask = val_mask_del
            test_mask = test_mask_del
            # 将列表转换为 numpy 数组
            train_mask = np.array(train_mask)
            val_mask = np.array(val_mask)
            test_mask = np.array(test_mask)
            # 使用 numpy 的 where() 函数找出所有值为True的索引
            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]
            test_idx = np.where(test_mask)[0]
            graph = g
        else:
            # 使用 numpy 的 where() 函数找出所有值为True的索引
            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]
            test_idx = np.where(test_mask)[0]
            graph = graph_full
    
                
        nfeat = graph.ndata.pop("feat").to(device)
        label = label.to(device)

        in_feats = nfeat.shape[1]
        n_classes = (label.max() + 1).item()
        # Create csr/coo/csc formats before launching sampling processes
        # This avoids creating certain formats in each data loader process, which saves momory and CPU.
        graph.create_formats_()
        # Pack data
        data = (
            train_idx,
            val_idx,
            test_idx,
            in_feats,
            label,
            n_classes,
            nfeat,
            graph,
            train_idx_full,
            val_idx_full,
            test_idx_full,
            graph_full
        )

        # Run 10 times
        test_accs = []
        for i in range(1):
            test_accs.append(run(args, device, data).cpu().numpy())
            print(
                "Average test accuracy:", np.mean(test_accs), "±", np.std(test_accs)
            )
        return np.mean(test_accs)
        
        
    # max_loss = [5,15,25,35,45,55,65]
    max_loss= [10,20,30,40,50,60]
    train_times = 5
    acc_total = np.reshape(np.arange(1,train_times+1), (1, train_times))    # acc矩阵
    for i in max_loss:
        acc_line = np.array([])     # acc矩阵的一行（一种part_num)
        for j in range(train_times):
            acc = multi_run(i)
            acc_line = np.append(acc_line, acc)
        acc_line = np.reshape(acc_line, (1, train_times))
        acc_total = np.append(acc_total, acc_line, axis=0)
    print(acc_total) 
    acc_final = np.mean(acc_total, axis=1)
    print(acc_final)