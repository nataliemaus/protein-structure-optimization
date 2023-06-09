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


def load_init_data_uniref(target_pdb_id, num_seqs_load=10_000):
    possible_score_filenames = glob.glob(f"../data/init_*_tmscores_{target_pdb_id}.csv")
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

    df_scores = pd.read_csv(filename_scores, header=None)
    scores = df_scores.values.squeeze()
    scores = scores[1:] # exclude first entry which is the one if seq score 
    train_y = torch.from_numpy(scores).float()
    train_y = train_y[0:num_seqs_load] 
    train_y = train_y.unsqueeze(-1) 
    uniref_seqs = load_uniref_seqs()
    train_x = uniref_seqs[0:num_seqs_load] 

    return train_x, train_y


def load_data_better(
    target_pdb_id,
    num_seqs_load=15_000,
): 
    probs_filename = f"../data/if_baseline_probs_human_{target_pdb_id}.csv"
    df = pd.read_csv(probs_filename)
    train_ys = df["tm_score"].values 
    probsh = df["prob_human"].values 
    train_xs = df["seq"].values.tolist() 

    # remove mask, etc. tokens occasionally output by ESM IF  
    train_xs = [x.replace("<mask>", "X") for x in train_xs]
    train_xs = [x.replace("<cls>", "X") for x in train_xs]
    train_xs = [x.replace("<sep>", "X") for x in train_xs]
    train_xs = [x.replace("<pad>", "X") for x in train_xs]
    train_xs = [x.replace("<eos>", "X") for x in train_xs]

    train_xs = np.array(train_xs)
    bool_arr = np.logical_not(np.isnan(train_ys))
    train_xs = train_xs[bool_arr]
    train_ys = train_ys[bool_arr]
    probsh = probsh[bool_arr]
    train_xs = train_xs.tolist() 
    if len(train_ys) < num_seqs_load: 
        print(f"WARNING: Number of Valid ESM IF baseline scores is {len(train_ys)} < {num_seqs_load}")
    
    train_y = torch.from_numpy(train_ys).float() 
    train_y = train_y.unsqueeze(-1) 
    train_y = train_y[0:num_seqs_load] 
    train_x = train_xs[0:num_seqs_load]
    probsh = probsh[0:num_seqs_load]

    return train_x, train_y, probsh


def load_init_data_esmif(
    target_pdb_id, 
    num_seqs_load=10_000,
):
    ''' Loads data from inverse fold baseline seqs 
    '''
    scores_filename_ = f"../data/if_baseline_tmscores_{target_pdb_id}_*.csv"
    possible_score_filenames = glob.glob(scores_filename_)

    train_xs = []
    train_ys = []
    for filename_scores in possible_score_filenames:
        df_scores = pd.read_csv(filename_scores, header=None)
        train_ys = train_ys + df_scores.values.squeeze().tolist() 
        wandb_run_name = filename_scores.split("/")[-1].split("_")[-1].split(".")[0]
        filename_seqs = f"../data/if_baseline_seqs_{target_pdb_id}_{wandb_run_name}.csv"
        df = pd.read_csv(filename_seqs, header=None)
        train_xs = train_xs + df.values.squeeze().tolist() 
    
    # remove mask, etc. tokens occasionally output by ESM IF  
    train_xs = [x.replace("<mask>", "X") for x in train_xs]
    train_xs = [x.replace("<cls>", "X") for x in train_xs]
    train_xs = [x.replace("<sep>", "X") for x in train_xs]
    train_xs = [x.replace("<pad>", "X") for x in train_xs]
    train_xs = [x.replace("<eos>", "X") for x in train_xs] 

    # filter out nan scores... 
    train_xs = np.array(train_xs)
    train_ys = np.array(train_ys)
    bool_arr = np.logical_not(np.isnan(train_ys))
    train_xs = train_xs[bool_arr]
    train_ys = train_ys[bool_arr]
    train_xs = train_xs.tolist() 
    if len(train_ys) < num_seqs_load: 
        print(f"WARNING: Number of Valid ESM IF baseline scores is {len(train_ys)} < {num_seqs_load}")
    
    train_y = torch.from_numpy(train_ys).float() 
    train_y = train_y.unsqueeze(-1) 
    train_y = train_y[0:num_seqs_load] 
    train_x = train_xs[0:num_seqs_load]
     
    return train_x, train_y


def load_init_data(
    target_pdb_id, 
    num_seqs_load=10_000,
    init_w_esmif=True
):
    if init_w_esmif:
        try:
            train_x, train_y, probsh = load_data_better(
                target_pdb_id,
                num_seqs_load=num_seqs_load,
            )
        except: 
            train_x, train_y = load_init_data_esmif(
                target_pdb_id, 
                num_seqs_load=num_seqs_load,
            )
            probsh = None 
    else:
        train_x, train_y = load_init_data_uniref(
            target_pdb_id, 
            num_seqs_load=num_seqs_load
        )
        probsh = None 
    return train_x, train_y, probsh


def create_data_v1(
    num_seqs=10,
    bsz=10,
    target_pdb_id="17_bp_sh3",
): 
    ''' Creates data using uniref seqs and a single esm if seq
    '''
    if_succeeded = True
    path_to_if_seqs = "../collected_pdbs/eval_all_results_new.csv"
    if_df = pd.read_csv(path_to_if_seqs)
    pdb_ids = if_df['pdb'].values.tolist() 
    if target_pdb_id in pdb_ids:
        target_idx = pdb_ids.index(target_pdb_id)
        seq0 = if_df['seq_0'].values.squeeze()[target_idx]
        seq1 = if_df['seq_1'].values.squeeze()[target_idx]
        seq2 = if_df['seq_2'].values.squeeze()[target_idx]
        if_seqs =[seq0, seq1, seq2]
    else:
        try:
            if_seq = inverse_fold(target_pdb_id=target_pdb_id, chain_id="A", model=None)
            if_seqs = [if_seq]  
        except:
            if_succeeded = False 
            if_seqs = []
    
    save_filename = f"../data/init_{num_seqs}_tmscores_{target_pdb_id}.csv"
    uniref_seqs = load_uniref_seqs()
    uniref_seqs = uniref_seqs[0:num_seqs-len(if_seqs)]
    objective = TMObjective(
        target_pdb_id=target_pdb_id,
        init_vae=False
    ) 

    seqs = if_seqs + uniref_seqs 

    all_scores = []
    for i in range(math.ceil(len(seqs)/bsz)):
        scores = objective.query_oracle(seqs[i*bsz:(i+1)*bsz])
        all_scores = all_scores + scores   

    if not if_succeeded:
        seqs = ["DUMMY"] + seqs 
        all_scores = [-1] + all_scores 

    all_scores = np.array(all_scores)
    pd.DataFrame(all_scores).to_csv(save_filename, index=None, header=None) 


def create_data_v2(
    num_seqs=10,
    bsz=10,
    target_pdb_id="17_bp_sh3",
    max_n_mutations=20,
): 
    ''' Creates data using random mutations of a few esm if seqs 
    '''
    path_to_if_seqs = "../collected_pdbs/eval_all_results_new.csv"
    if_df = pd.read_csv(path_to_if_seqs)
    pdb_ids = if_df['pdb'].values.tolist() 
    if target_pdb_id in pdb_ids:
        target_idx = pdb_ids.index(target_pdb_id)
        seq0 = if_df['seq_0'].values.squeeze()[target_idx]
        seq0 = seq0.replace("'", "")
        seq1 = if_df['seq_1'].values.squeeze()[target_idx]
        seq1 = seq1.replace("'", "")
        seq2 = if_df['seq_2'].values.squeeze()[target_idx]
        seq2 = seq2.replace("'", "")
        if_seqs =[seq0, seq1, seq2] 
    else:
        seq0 = inverse_fold(target_pdb_id=target_pdb_id, chain_id="A", model=None)
        if_seqs = [seq0] 
    
    scores_filename = f"../data/init_{num_seqs}_tmscores_V2_{target_pdb_id}.csv"
    seqs_filename = f"../data/init_{num_seqs}_V2_seqs_{target_pdb_id}.csv"
    objective = TMObjective(
        target_pdb_id=target_pdb_id,
        init_vae=False,
    ) 

    seqs = copy.deepcopy(if_seqs)
    scores = []
    for _ in range(num_seqs - len(if_seqs)):
        if_seq = random.choice(if_seqs) 
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
    parser.add_argument('--num_seqs', type=int, default=1000 ) 
    parser.add_argument('--bsz', type=int, default=10 ) 
    parser.add_argument('--target_pdb_id', default="17_bp_sh3" ) 
    parser.add_argument('--data_version', type=int, default=1 ) 
    parser.add_argument('--max_n_mutations', type=int, default=10 ) # only releant for v2

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
