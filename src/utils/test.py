import time
import numpy as np
import torch

n = 100

LD = [{'a': np.ones((n,n)), 'b': 2*np.ones((n,n))}, {'a': 3*np.ones((n,n)), 'b': 4*np.ones((n,n))}]

start = time.time()

v = {k: np.stack([dic[k] for dic in LD]) for k in LD[0]}

print(time.time()-start)
print(v["a"])
print(torch.from_numpy(v["a"]))
