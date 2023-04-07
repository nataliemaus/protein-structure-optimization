import sys 
sys.path.append("../")
import argparse 
import pandas as pd 
import numpy as np 
import math 
from lolbo.tm_objective import TMObjective
import glob 
import torch 
from oracle.fold import inverse_fold
import copy
import random 
from constants import ALL_AMINO_ACIDS

def load_uniref_seqs():
    path = "../uniref_vae/uniref-small.csv"
    df = pd.read_csv(path)
    seqs = df['sequence'].values
    return seqs.tolist() 

def load_uniref_scores(target_pdb_id, num_seqs_load=10_000):
    possible_filenames = glob.glob(f"../data/init_*_tmscores_{target_pdb_id}.csv")
    nums_seqs = []
    for filename in possible_filenames:
        n_seqs = int(filename.split("/")[-1].split("_")[1])
        nums_seqs.append(n_seqs)
    nums_seqs = np.array(nums_seqs)
    max_n_seqs = nums_seqs.max() 
    if max_n_seqs < num_seqs_load:
        print(f"Have not saved enough initilization data to load {num_seqs_load} seqs")
        assert 0 
    filename_scores = possible_filenames[np.argmax(nums_seqs)]
    df = pd.read_csv(filename_scores, header=None)
    train_y = torch.from_numpy(df.values.squeeze()).float()
    train_y = train_y[0:num_seqs_load] 
    return train_y.unsqueeze(-1) 


def load_init_data(target_pdb_id, num_seqs_load=10_000):
    possible_score_filenames = glob.glob(f"../data/init_*_tmscores_V2_{target_pdb_id}.csv")
    nums_seqs = []
    for filename in possible_score_filenames:
        n_seqs = int(filename.split("/")[-1].split("_")[1])
        nums_seqs.append(n_seqs)
    nums_seqs = np.array(nums_seqs)
    max_n_seqs = nums_seqs.max() 
    if max_n_seqs < num_seqs_load:
        print(f"Have not saved enough initilization data to load {num_seqs_load} seqs")
        assert 0 
    filename_scores = possible_score_filenames[np.argmax(nums_seqs)]
    filename_seqs = f"../data/init_{max_n_seqs}_V2_seqs_{target_pdb_id}.csv"

    df_scores = pd.read_csv(filename_scores, header=None)
    train_y = torch.from_numpy(df_scores.values.squeeze()).float()
    train_y = train_y[0:num_seqs_load] 
    train_y = train_y.unsqueeze(-1) 

    df = pd.read_csv(filename_seqs)
    train_x = df.values.squeeze().tolist() 
    train_x = train_x[0:num_seqs_load] 

    return train_x, train_y


def create_data_v1(
    num_seqs=10,
    bsz=10,
    target_pdb_id="17_bp_sh3",
): 
    if_seq = inverse_fold(target_pdb_id=target_pdb_id, chain_id="A", model=None)
    save_filename = f"../data/init_{num_seqs}_tmscores_{target_pdb_id}.csv"
    seqs = load_uniref_seqs()
    objective = TMObjective(
        target_pdb_id=target_pdb_id,
    ) 
    seqs = seqs[0:num_seqs-1]
    seqs = [if_seq] + seqs 

    all_scores = []
    for i in range(math.ceil(num_seqs/bsz)):
        scores = objective.query_oracle(seqs[i*bsz:(i+1)*bsz])
        all_scores = all_scores + scores   

    all_scores = np.array(all_scores)
    pd.DataFrame(all_scores).to_csv(save_filename, index=None, header=None) 


def create_data_v2(
    num_seqs=10,
    bsz=10,
    target_pdb_id="17_bp_sh3",
    max_n_mutations=20,
): 
    if_seq = inverse_fold(target_pdb_id=target_pdb_id, chain_id="A", model=None)
    scores_filename = f"../data/init_{num_seqs}_tmscores_V2_{target_pdb_id}.csv"
    seqs_filename = f"../data/init_{num_seqs}_V2_seqs_{target_pdb_id}.csv"
    objective = TMObjective(
        target_pdb_id=target_pdb_id,
    ) 

    seqs = [if_seq]
    scores = []
    for _ in range(num_seqs - 1):
        new_seq = copy.deepcopy(if_seq)
        new_seq = [char for char in new_seq] 
        random_mutation_idx = random.randint(0, len(if_seq) - 1)
        num_mutations = random.randint(1,max_n_mutations)
        for _ in range(num_mutations):
            new_aa = random.choice(ALL_AMINO_ACIDS)
            new_seq[random_mutation_idx] = new_aa 
        new_seq = "".join(new_seq)
        seqs.append(new_seq) 
    
    all_scores = []
    for i in range(math.ceil(num_seqs/bsz)):
        scores = objective.query_oracle(seqs[i*bsz:(i+1)*bsz])
        all_scores = all_scores + scores   

    pd.DataFrame(np.array(all_scores)).to_csv(scores_filename, index=None, header=None) 
    pd.DataFrame(np.array(seqs)).to_csv(seqs_filename, index=None, header=None)



if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--num_seqs', type=int, default=10_000 ) 
    parser.add_argument('--bsz', type=int, default=10 ) 
    parser.add_argument('--target_pdb_id', default="17_bp_sh3" ) 
    parser.add_argument('--data_version', type=int, default=2 ) 
    parser.add_argument('--max_n_mutations', type=int, default=10 )

    args = parser.parse_args() 
    if args.data_version == 1:
        create_data_v1(
            num_seqs=args.num_seqs,
            bsz=args.bsz,
            target_pdb_id=args.target_pdb_id,
        )
    elif args.data_version == 2:
        create_data_v2(
            num_seqs=args.num_seqs,
            bsz=args.bsz,
            target_pdb_id=args.target_pdb_id,
            max_n_mutations=args.max_n_mutations,
        )

    
    

        
        
        
