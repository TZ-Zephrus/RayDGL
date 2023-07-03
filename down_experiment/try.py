import argparse
import os
import sys
import time
import dgl
import numpy as np
import torch as th
from dgl.data import CoraGraphDataset
from igb import download
import argparse, dgl
from igb.dataloader import IGB260MDGLDataset
# download.download_dataset(path='/home/asd/文档/wtz/wtz/RayDGL/dataset/igb-small', dataset_type='homogeneous', dataset_size='small')

def igbdataset():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--path', type=str, default='/home/asd/文档/wtz/wtz/RayDGL/dataset/igb-small', 
    #     help='path containing the datasets')
    # parser.add_argument('--dataset_size', type=str, default='small',
    #     choices=['tiny', 'small', 'medium', 'large', 'full'], 
    #     help='size of the datasets')
    # parser.add_argument('--num_classes', type=int, default=19, 
    #     choices=[19, 2983], help='number of classes')
    # parser.add_argument('--in_memory', type=int, default=1, 
    #     choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
    # parser.add_argument('--synthetic', type=int, default=0,
    #     choices=[0, 1], help='0:nlp-node embeddings, 1:random')
    # args = parser.parse_args()
    dataset = IGB260MDGLDataset(path='/home/asd/文档/wtz/wtz/RayDGL/dataset/igb-small', dataset_size='small', num_classes=19, in_memory=1, synthetic=0)
    return dataset
graph = igbdataset()[0]
print(graph)

