# 画直线或矩形
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
merge = nn.Conv1d(64,64,kernel_size=1)
proj = nn.ModuleList([deepcopy(merge) for _ in range(3)])
query = torch.ones((1,256,384))
key = torch.ones((1,256,384))
value = torch.ones((1,256,384))
# for l,x in zip(proj,(query,key,value)):
#     print(l(x).view(1,64,4,384))
#     print(x)


q = np.arange(0,24).reshape(1,2,3,4)
k = np.arange(10,40).reshape(1,2,3,5)
c = np.einsum('')