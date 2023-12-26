import numpy as np
import dgl
from dgl.data import CoraGraphDataset, RedditDataset
import torch
# import pandas as pd


from collections import Counter

# 示例列表
my_list = [1, 2, 3, 4, 5]

# 使用完整切片获取整个列表
new_list = my_list[1:2:-1]

print(new_list)  # 输出: [1, 2, 3, 4, 5]