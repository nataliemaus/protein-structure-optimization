import sys 
sys.path.append("../")
# from oracle.fold import inverse_fold_many_seqs
import wandb 
import os
os.environ["WANDB_SILENT"] = "True"
import pandas as pd 
from lolbo.tm_objective import TMObjective
import esm 
import numpy as np 
import argparse 
import torch 
import os 
# os.environ["CUDA_VISIBLE_DEVICES"]="7"


def create_wandb_tracker(
    config_dict,
    wandb_project_name,
    wandb_entity="nmaus",
):
    tracker = wandb.init(
        project=wandb_project_name,
        entity=wandb_entity,
        config=config_dict,
    ) 
    return tracker 

    
# self.wandb_project_name = f"optimimze-{self.task_id}"

@torch.no_grad()
def run_if_baseline(
    max_n_oracle_calls=500_000_000,
    bsz=10,
    target_pdb_id="17_bp_sh3",
    save_freq=10,
    tracker=None,
    n_init=1_000,
): 
    device = "cuda:0"
    wandb_run_name = wandb.run.name 
    scores_filename = f"../data/if_baseline_tmscores_{target_pdb_id}_{wandb_run_name}.csv"
    seqs_filename = f"../data/if_baseline_seqs_{target_pdb_id}_{wandb_run_name}.csv"
    objective = TMObjective(
        target_pdb_id=target_pdb_id,
        init_vae=False,
    ) 
    if_model, _ = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    if_model = if_model.eval().to(device)
    
    # if_model = if_model.eval()
    pdb_path = f"../oracle/target_cif_files/{target_pdb_id}.cif" 
    structure = esm.inverse_folding.util.load_structure(pdb_path, "A")
    coords, _ = esm.inverse_folding.util.extract_coords_from_structure(structure)
    # if_model = if_model.cuda() 

    
    # try:
    #     seqs = if_model.sample(coords, temperature=1, num_seqs=10)
    #           num_seqs is an unrecognized argument... 
    #     scores = objective.query_oracle(seqs)
    # except:
    #     import pdb 
    #     pdb.set_trace()  
    
    # print('SUCCESS!!!')
    # import pdb 
    # pdb.set_trace()  

    # try: 
    #     coords2 = torch.from_numpy(coords).cuda() 
    # except:
    #     import pdb 
    #     pdb.set_trace() 

    # get n_init seqs and scores 
    seqs = []
    scores = []
    for _ in range(n_init):
        sampled_seq = if_model.sample(coords, temperature=1, device=device) 
        seqs.append(sampled_seq)
        score = objective.query_oracle([sampled_seq])[0]
        if np.isnan(score):
            score = -1 
        scores.append(score) 

    best_idx = np.argmax(np.array(scores))
    best_score = scores[best_idx]
    best_seq = seqs[best_idx] 

    steps = 0 
    num_calls = 0
    while num_calls < max_n_oracle_calls:
        seqs_batch = []
        for _ in range(bsz):
            sampled_seq = if_model.sample(coords, temperature=1, device=device) 
            seqs_batch.append(sampled_seq)

        scores_batch = objective.query_oracle(seqs_batch)

        # catch nans and replace with -1 ... 
        scores_batch = np.array(scores_batch)
        scores_batch[np.isnan(scores_batch)] = -1 # replace nan w/ -1 
        scores_batch = scores_batch.tolist() 

        num_calls += len(scores_batch) 
        seqs = seqs + seqs_batch 
        scores = scores + scores_batch 
        best_idx_in_batch = np.argmax(np.array(scores_batch))
        # best_in_batch = np.array(scores_batch).max()
        best_score_in_batch = scores_batch[best_idx_in_batch]
        best_seq_in_batch = seqs_batch[best_idx_in_batch] 

        if best_score_in_batch > best_score: 
            best_score = best_score_in_batch
            best_seq = best_seq_in_batch
        tracker.log({
            "best_found":best_score,
            "n_oracle_calls":num_calls,
            "best_input_seen":best_seq,
        }) 
        if (steps % save_freq == 0) or (steps in [0, 10, 100, 1_000]):
            pd.DataFrame(np.array(scores)).to_csv(scores_filename, index=None, header=None) 
            pd.DataFrame(np.array(seqs)).to_csv(seqs_filename, index=None, header=None)
        steps += 1

    pd.DataFrame(np.array(scores)).to_csv(scores_filename, index=None, header=None) 
    pd.DataFrame(np.array(seqs)).to_csv(seqs_filename, index=None, header=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--max_n_oracle_calls', type=int, default=150_000 ) 
    parser.add_argument('--bsz', type=int, default=10 ) 
    parser.add_argument('--save_freq', type=int, default=1_000_000_000_000 ) 
    parser.add_argument('--n_init', type=int, default=1_000 ) 
    parser.add_argument('--if_baseline', type=bool, default=True )
    parser.add_argument('--target_pdb_id', default="17_bp_sh3" ) 

    args = parser.parse_args() 
    tracker = create_wandb_tracker(
        config_dict=vars(args),
        wandb_project_name="optimimze-tm",
        wandb_entity="nmaus",
    )
    run_if_baseline(
        max_n_oracle_calls=args.max_n_oracle_calls,
        bsz=args.bsz,
        target_pdb_id=args.target_pdb_id,
        save_freq=args.save_freq,
        tracker=tracker,
        n_init=args.n_init,
    )

    # CUDA_VISIBLE_DEVICES=0 python3 if_baseline.py --target_pdb_id sample587 
