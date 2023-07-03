import argparse
import os
import sys
import time
import dgl
import numpy as np
import torch as th
from dgl.data import CoraGraphDataset, PubmedGraphDataset, FlickrDataset, RedditDataset
from igb.dataloader import IGB260MDGLDataset

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
        default="/root/wtz/RayDGL/dataset/pubmed_random 20 partition",
        help="Output path of partitioned graph.",
    )
    args1 = argparser.parse_args()


    start = time.time()
    datasetname = 'igb-small'
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
    if datasetname == 'flickr':
        dataset = FlickrDataset(raw_dir='/home/asd/文档/wtz/wtz/RayDGL/dataset/{}'.format(datasetname))
        num_class = 7
    if datasetname == 'igb_small':
        parser = argparse.ArgumentParser()
        parser.add_argument('--path', type=str, default='/home/asd/文档/wtz/wtz/RayDGL/dataset/igb_small', 
            help='path containing the datasets')
        parser.add_argument('--dataset_size', type=str, default='small',
            choices=['tiny', 'small', 'medium', 'large', 'full'], 
            help='size of the datasets')
        parser.add_argument('--num_classes', type=int, default=19, 
            choices=[19, 2983], help='number of classes')
        parser.add_argument('--in_memory', type=int, default=1, 
            choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
        parser.add_argument('--synthetic', type=int, default=0,
            choices=[0, 1], help='0:nlp-node embeddings, 1:random')
        args2 = parser.parse_args()
        dataset = IGB260MDGLDataset(args2)
        num_class = 19
    
    
    g = dataset[0]
    num_parts = 100
    g = dgl.remove_self_loop(g)  # 消除自环
    print(
        "load {} takes {:.3f} seconds".format(datasetname, time.time() - start)
    )
    print("|V|={}, |E|={}".format(g.num_nodes(), g.num_edges()))
    print(
        "train: {}, valid: {}, test: {}".format(
            th.sum(g.ndata["train_mask"]),
            th.sum(g.ndata["val_mask"]),
            th.sum(g.ndata["test_mask"]),
        )
    )
    if args1.balance_train:
        balance_ntypes = g.ndata["train_mask"]
    else:
        balance_ntypes = None

    if args1.undirected:
        sym_g = dgl.to_bidirected(g, readonly=True)
        for key in g.ndata:
            sym_g.ndata[key] = g.ndata[key]
        g = sym_g

    dgl.distributed.partition.partition_graph(
        g,
        datasetname,
        num_parts,
        '/home/asd/文档/wtz/wtz/RayDGL/dataset/{} {} partition'.format(datasetname, num_parts),
        num_hops=1,
        part_method='metis',
        # balance_ntypes=g.ndata['train_mask']
    )