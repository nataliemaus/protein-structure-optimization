# uniref-small.csv 

import pandas as pd
import numpy as np

df = pd.read_csv("uniref-small.csv")
import pdb 
pdb.set_trace() 
seqs = df['sequence']

lens = [len(s) for s in seqs]
lens = np.array(lens)
print("min length", lens.min(), "max length", lens.max())
# min length 100 max length 299

print("total n seqs", len(seqs))
# total n seqs 1,500,000 