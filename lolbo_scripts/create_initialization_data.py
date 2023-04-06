import sys 
sys.path.append("../")
import argparse 
import pandas as pd 
import numpy as np 
import math 
from lolbo.tm_objective import TMObjective
import glob 
import torch 

def load_uniref_seqs():
    path = "../uniref_vae/uniref-small.csv"
    df = pd.read_csv(path)
    seqs = df['sequence'].values
    return seqs.tolist() 

def load_uniref_scores(target_pdb_id, n_seqs=10_000):
    possible_filenames = glob.glob(f"../data/init_*_tmscores_{target_pdb_id}.csv")
    nums_seqs = []
    for filename in possible_filenames:
        n_seqs = int(filename.split("/")[-1].split("_")[1])
        nums_seqs.append(n_seqs)
    nums_seqs = np.array(nums_seqs)
    max_n_seqs = nums_seqs.max() 
    if max_n_seqs < n_seqs:
        print(f"Have not saved enough initilization data to load {n_seqs} seqs")
        assert 0 
    filename_scores = possible_filenames[np.argmax(nums_seqs)]
    df = pd.read_csv(filename_scores, header=None)
    train_y = torch.from_numpy(df.values).float()
    import pdb 
    pdb.set_trace() 
    return train_y.unsqueeze(-1) 


def main(
    num_seqs=10,
    bsz=10,
    target_pdb_id="17_bp_sh3",
): 
    save_filename = f"../data/init_{num_seqs}_tmscores_{target_pdb_id}.csv"
    seqs = load_uniref_seqs()
    objective = TMObjective(
        target_pdb_id=target_pdb_id,
    )
    seqs = seqs[0:num_seqs]

    all_scores = []
    for i in range(math.ceil(num_seqs/bsz)):
        scores = objective.query_oracle(seqs[i*bsz:(i+1)*bsz])
        all_scores = all_scores + scores   

    all_scores = np.array(all_scores)
    pd.DataFrame(all_scores).to_csv(save_filename, index=None, header=None) 



if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--num_seqs', type=int, default=10_000 ) 
    parser.add_argument('--bsz', type=int, default=10 ) 
    parser.add_argument('--target_pdb_id', default="17_bp_sh3" ) 
    args = parser.parse_args() 
    main(
        num_seqs=args.num_seqs,
        bsz=args.bsz,
        target_pdb_id=args.target_pdb_id,
    )

    
    

        
        
        
