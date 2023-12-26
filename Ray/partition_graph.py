import argparse
import os
import sys
import time
import dgl
import numpy as np
import torch as th
from dgl.data import CoraGraphDataset, CoraFullDataset, PubmedGraphDataset, RedditDataset, FraudDataset
from ogb.nodeproppred import DglNodePropPredDataset

# sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="datasets: reddit, ogb-product, ogb-paper100M",
    )
    argparser.add_argument(
        "--num_parts", type=int, default=3, help="number of partitions"
    )
    argparser.add_argument(
        "--part_method", type=str, default="metis", help="the partition method"
    )
    argparser.add_argument(
        "--balance_train",
        default=True, 
        action="store_true",
        help="balance the training size in each partition.",
    )
    argparser.add_argument(
        "--undirected",
        action="store_true",
        help="turn the graph into an undirected graph.",
    )
    argparser.add_argument(
        "--balance_edges",
        action="store_true",
        help="balance the number of edges in each partition.",
    )
    argparser.add_argument(
        "--num_trainers_per_machine",
        type=int,
        default=1,
        help="the number of trainers per machine. The trainer ids are stored\
                                in the node feature 'trainer_id'",
    )
    argparser.add_argument(
        "--output",
        type=str,
        default="/root/wtz/RayDGL/dataset/cora partition additional",
        help="Output path of partitioned graph.",
    )
    args = argparser.parse_args()

    start = time.time()
    
    # # ogbn dataset
    # dataset = DglNodePropPredDataset(name = "ogbn-arxiv", root = "/root/wtz/RayDGL/dataset/ogbn-arxiv")
    # g, labels = dataset[0]
    # split_idx = dataset.get_idx_split()
    # train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    # graph, label = dataset[0] # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
    # labels = labels[:, 0]
    # g.ndata['labels'] = labels
    
    # cora dataset
    datasetname = 'reddit'
    num_parts = 100

    if datasetname == 'cora':
        dataset = CoraGraphDataset(raw_dir='/root/wtz/RayDGL/dataset/{}'.format(datasetname))
        num_class = 7
    if datasetname == 'cora_nobalance':
        dataset = CoraGraphDataset(raw_dir='/root/wtz/RayDGL/dataset/{}'.format(datasetname))
        num_class = 7
    if datasetname == 'pubmed':
        dataset = PubmedGraphDataset(raw_dir='/root/wtz/RayDGL/dataset/{}'.format(datasetname))
        num_class = 3
    if datasetname == 'reddit':
        dataset = RedditDataset(raw_dir='/root/wtz/RayDGL/dataset/{}'.format(datasetname))
        num_class = 41
    
    
    g = dataset[0]
    # g = dgl.remove_self_loop(g)  # 消除自环
    
    print(g.ndata)


    print(
        "load {} takes {:.3f} seconds".format(args.dataset, time.time() - start)
    )
    print("|V|={}, |E|={}".format(g.num_nodes(), g.num_edges()))
    print(
        "train: {}, valid: {}, test: {}".format(
            th.sum(g.ndata["train_mask"]),
            th.sum(g.ndata["val_mask"]),
            th.sum(g.ndata["test_mask"]),
        )
    )
    if args.balance_train:
        balance_ntypes = g.ndata["train_mask"]
    else:
        balance_ntypes = None

    if args.undirected:
        sym_g = dgl.to_bidirected(g, readonly=True)
        for key in g.ndata:
            sym_g.ndata[key] = g.ndata[key]
        g = sym_g

    
    node_map, edge_map = dgl.distributed.partition.partition_graph(
        g,
        graph_name = datasetname,
        num_parts = 100,
        out_path = "/root/wtz/RayDGL/dataset/{} {} partition".format(datasetname, num_parts),
        num_hops = 1, 
        part_method=args.part_method,
        # balance_ntypes=g.ndata['train_mask'],
        return_mapping=True
    )
    print(node_map)
    print(edge_map)
