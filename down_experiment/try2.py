from ogb.nodeproppred import DglNodePropPredDataset
import numpy as np
import dgl
from dgl.data import CoraGraphDataset, RedditDataset
import torch
import pandas as pd


a = np.array([0,1,2,3,4,5,6])
b = np.array([3,4,1,4,7,3,8])
c = np.array([0,1,2])

def del_edge4(target_node, del_node):    # 从一个里面删另一个里有的
    target = pd.Series(target_node)
    mask_del = target.isin(del_node)
    target_node_new = np.array(target[~mask_del])
    return target_node_new

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

q = [True, False, True,True,True,False,True,False]
p = [0,1,2,4,5]
e, f = del_edge3(a,b,c)
d = del_edge4(b,c)
print(d)
print(e,f)
print(del_mask(p,q))