import argparse
import os
import sys
import time
import dgl
import numpy as np
import torch as th
from dgl.data import CoraGraphDataset
np.set_printoptions(threshold=np.inf)

a = np.array([[1,2.3,3,43.2,50.5],
          [2,3,4,5,8]])
print(np.median(a,axis=1))