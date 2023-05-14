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
import glob 
# os.environ["CUDA_VISIBLE_DEVICES"]="7"
import time 
from oracle.get_prob_human import get_probs_human, load_human_classier_model
import math 
from oracle.edit_distance import compute_edit_distance


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

def load_existing_esmif_data(
    target_pdb_id, 
):
    ''' Loads data from inverse fold baseline seqs 
    '''
    scores_filename_ = f"../data/if_baseline_tmscores_{target_pdb_id}_*.csv"
    possible_score_filenames = glob.glob(scores_filename_)

    if len(possible_score_filenames) == 0: # if none computed yet 
        return [], []

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

    seqs = train_xs.tolist() 
    scores = train_ys.tolist() 
    return seqs, scores 


def compute_and_save_if_baseline_human_probs(
    the_target_pdb_id="all"
):
    BSZ = 32 
    human_classifier_tokenizer,  human_classifier_model  = load_human_classier_model()
    if the_target_pdb_id == "all":
        target_pdb_ids = glob.glob("../data/if_baseline_tmscores_*sample*.csv")
        target_pdb_ids = [filename.split("/")[-1].split("_")[-2] for filename in target_pdb_ids]
        print("Target pdb ids:", target_pdb_ids) 
    else:
        target_pdb_ids = [the_target_pdb_id] 
    for target_pdb_id in target_pdb_ids:
        seqs, scores = load_existing_esmif_data(target_pdb_id) 
        n_sub_batches = math.ceil(len(seqs)/BSZ)
        probs_h = [] 
        for i in range(n_sub_batches):
            probsh_tensor = get_probs_human(
                seqs_list=seqs[i*BSZ : (i+1)*BSZ], 
                human_tokenizer=human_classifier_tokenizer, 
                human_model=human_classifier_model
            )
            probs_h = probs_h + probsh_tensor.tolist() 


        # probs_h = [] 
        # for seq in seqs:
            # probh = get_probs_human(
            #     seq=seq, 
            #     human_tokenizer=human_classifier_tokenizer, 
            #     human_model=human_classifier_model,
            # )
            # probs_h.append(probh)
        max_prob_h = np.array(probs_h).max() 
        print(f"for target {target_pdb_id}, max prob human = {max_prob_h}")
        probs_filename = f"../data/if_baseline_probs_human_{target_pdb_id}.csv"
        data = {
            "seq":seqs, 
            "tm_score":scores,
            "prob_human":probs_h
        } 
        df = pd.DataFrame.from_dict(data)
        df.to_csv(probs_filename, index=None) 


def log_if_baseline_constrained(target_pdb_id, min_prob_human):
        probs_filename = f"../data/if_baseline_probs_human_{target_pdb_id}.csv"
        df = pd.read_csv(probs_filename)
        tm_scores = df["tm_score"].values 
        probsh = df["prob_human"].values 
        seqs = df["seq"].values 

        args_dict = {
            "max_n_oracle_calls":150_000,
            "n_init":1_000,
            "if_baseline":True,
            "target_pdb_id":target_pdb_id,
            "min_prob_human":min_prob_human,
        }

        tracker = create_wandb_tracker(
            config_dict=args_dict,
            wandb_project_name="optimimze-tm",
            wandb_entity="nmaus",
        )
        init_scores = tm_scores[0:1_000]
        init_seqs = seqs[0:1_000]
        init_probsh = probsh[0:1_000]
        valid_init_scores = init_scores[init_probsh >= min_prob_human]
        valid_init_seqs = init_seqs[init_probsh >= min_prob_human]
        if len(valid_init_scores) > 0: # if any number of valid init scores 
            best_found =  valid_init_scores.max() 
            best_input_seen = valid_init_seqs[valid_init_scores.argmax()] 
        else:
            best_found = 0.0
            best_input_seen = "" 
        
        remaining_scores = tm_scores[1_000:]
        remaining_seqs = seqs[1_000:]
        remaining_probsh = probsh[1_000:] 
        n_oracle_calls = 0
        for ix, tm_score in enumerate(remaining_scores):
            if best_found > 0:
                tracker.log({
                    "best_found":best_found,
                    "best_input_seen":best_input_seen,
                    "n_oracle_calls":n_oracle_calls
                }) 
            if (remaining_probsh[ix] >= min_prob_human) and (tm_score > best_found):
                best_found = tm_score 
                best_input_seen = remaining_seqs[ix]
            n_oracle_calls += 1
        
        tracker.finish() 

            

def log_if_baseline_robot(
    target_pdb_id, 
    M,
    tau,
    step_size=100,
):
        # probs_filename = f"../data/if_baseline_probs_human_{target_pdb_id}.csv"
        # df = pd.read_csv(probs_filename)
        # tm_scores = df["tm_score"].values 
        # # probsh = df["prob_human"].values 
        # seqs = df["seq"].values 

        seqs, tm_scores = load_existing_esmif_data(target_pdb_id) 

        args_dict = {
            "M":M,
            "tau":tau,
            "max_n_oracle_calls":150_000,
            "num_initialization_points":1_000,
            "if_baseline":True, 
            "target_pdb_id":target_pdb_id,
        } 
        # compute_edit_distance 

        tracker = create_wandb_tracker(
            config_dict=args_dict,
            wandb_project_name="ROBOT-tm",
            wandb_entity="nmaus",
        )


        def is_feasible(x, higher_ranked_xs): 
            for higher_ranked_x in higher_ranked_xs:
                if compute_edit_distance(x, higher_ranked_x) < tau:
                    return False 
            return True 

        def get_top_m_scores_and_seqs(scores_tensor, seqs_list_):
            M_diverse_scores = []
            tr_center_xs = []
            idx_num = 0
            _, top_t_idxs = torch.topk(scores_tensor, len(scores_tensor))
            for _ in range(M):
                while True: 
                    # if we run out of feasible points in dataset
                    if idx_num >= len(scores_tensor): 
                        print("out of feasible points")
                        return None, None
                    # otherwise, finding highest scoring feassible point in remaining dataset for tr center
                    center_idx = top_t_idxs[idx_num]
                    center_score = scores_tensor[center_idx].item()
                    center_x = seqs_list_[center_idx]
                    idx_num += 1
                    if is_feasible(center_x, higher_ranked_xs=tr_center_xs):
                        break 

                tr_center_xs.append(center_x) 
                M_diverse_scores.append(center_score)

            M_diverse_scores = np.array(M_diverse_scores)
            M_diverse_xs = tr_center_xs
            return M_diverse_scores, M_diverse_xs

        
        # remaining_scores = tm_scores[1_000:]
        # remaining_seqs = seqs[1_000:]
        n_oracle_calls = 0
        # for ix, tm_score in enumerate(remaining_scores):
        for i in range(1_000, len(tm_scores), step_size):
            M_diverse_scores, M_diverse_xs = get_top_m_scores_and_seqs(
                scores_tensor=torch.tensor(tm_scores[0:i]).float(), 
                seqs_list_=seqs[0:i],
            ) 
            if M_diverse_scores is None:
                break # if none found 
            tracker.log({
                "mean_score_diverse_set":M_diverse_scores.mean(),
                "min_score_diverse_set":M_diverse_scores.min(),
                "max_score_diverse_set":M_diverse_scores.max(),
                "n_oracle_calls":n_oracle_calls
            }) 
            n_oracle_calls += step_size

        tracker.log({"M_diverse_xs":M_diverse_xs})
        tracker.finish() 






def analyze_probs_human():
    thresholds = [
        0.999, 0.998, 0.997, 0.996, 
        0.995, 0.994, 0.993, 0.992, 0.991, 0.990, 
        0.989, 0.988, 0.987, 0.986, 
        0.985, 0.984, 0.983, 0.982, 0.981, 0.980, 
        0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 
        0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1
    ] 
    # sample228: 0.990 
    # sample479: 0.4, 0.95=NONE 
    # just pick a couple good example proteins and set threshold to 
    # where ESM IF GETS LITERALLY NONE... Or does abysmally 
    target_pdb_ids = glob.glob("../data/if_baseline_probs_human_*.csv")
    target_pdb_ids = [filename.split("/")[-1].split("_")[-1].split(".")[0] for filename in target_pdb_ids]
    print("Target pdb ids:", target_pdb_ids) 
    for target_pdb_id in target_pdb_ids:
        print(f"\ntarget_pdb_id: {target_pdb_id}")
        probs_filename = f"../data/if_baseline_probs_human_{target_pdb_id}.csv"
        df = pd.read_csv(probs_filename)
        # seqs = df["seq"] 
        scores = df["tm_score"]
        probs_h = df["prob_human"]
        print(f"total n scores: {len(scores)}, max score: {scores.max()}") 
        for threshold in thresholds:
            thresholded_scores = scores[probs_h >= threshold]
            print(f"max score w/ threshold prob human {threshold}: {thresholded_scores.max()}")

        # import pdb 
        # pdb.set_trace() 

# Threshold 0.9 Poor Preformers
# 1. 587 
# 2. 611 
# 3. 286


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

    # get n_init seqs and scores 
    seqs, scores = load_existing_esmif_data(target_pdb_id)
    # seqs = []
    # scores = []
    if_time_sum = 0.0 
    if_n_timed = 0 
    query_time_sum = 0.0 
    query_n_timed = 0 
    n_precomputed = len(scores)
    if n_precomputed < n_init:
        for _ in range(n_init - n_precomputed):
            start = time.time() 
            sampled_seq = if_model.sample(coords, temperature=1, device=device) 
            if_time = time.time() - start 
            tracker.log({"if_time":if_time})
            if_time_sum += if_time 
            if_n_timed += 1 
            tracker.log({"if_time_running_avg":if_time_sum/if_n_timed})
            seqs.append(sampled_seq)
            start = time.time() 
            score = objective.query_oracle([sampled_seq])[0]
            query_time = time.time() - start 
            tracker.log({"query_time":query_time})
            query_time_sum += query_time 
            query_n_timed += 1 
            tracker.log({"query_time_running_avg":query_time_sum/query_n_timed})
            if np.isnan(score):
                score = -1 
            scores.append(score) 

    best_idx = np.argmax(np.array(scores))
    best_score = scores[best_idx]
    best_seq = seqs[best_idx] 

    steps = 0 
    if n_precomputed > n_init:
        num_calls = n_precomputed - n_init # might start at higher num calls (continue prev runs...)
    else:
        num_calls = 0 
    while num_calls < max_n_oracle_calls:
        seqs_batch = []
        for _ in range(bsz):
            start = time.time() 
            sampled_seq = if_model.sample(coords, temperature=1, device=device) 
            if_time = time.time() - start 
            tracker.log({"if_time":if_time})
            if_time_sum += if_time 
            if_n_timed += 1 
            tracker.log({"if_time_running_avg":if_time_sum/if_n_timed})
            seqs_batch.append(sampled_seq)

        start = time.time() 
        scores_batch = objective.query_oracle(seqs_batch)
        query_time = (time.time() - start)/len(seqs_batch) 
        tracker.log({"query_time":query_time}) 
        query_time_sum += query_time 
        query_n_timed += 1 
        tracker.log({"query_time_running_avg":query_time_sum/query_n_timed})

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
        if (steps % save_freq == 0) or (steps in [0, 10, 100, 1_000, 10_000]):
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
    parser.add_argument('--analyze_probs_human', type=bool, default=False ) # meh 
    parser.add_argument('--compute_probs_h', type=bool, default=False )
    parser.add_argument('--log_if_baseline_constrained', type=bool, default=False )
    parser.add_argument('--min_prob_human', type=float, default=0.8 ) 

    parser.add_argument('--M', type=int, default=10 ) 
    parser.add_argument('--tau', type=int, default=10 ) 
    parser.add_argument('--step_size', type=int, default=100 ) 
    parser.add_argument('--log_if_baseline_robot', type=bool, default=False )

    parser.add_argument('--all_robot', type=bool, default=False )

    args = parser.parse_args() 

    # CUDA_VISIBLE_DEVICES=1 python3 if_baseline.py --all_robot True 

    # python3 if_baseline.py --target_pdb_id sampleXX --max_n_oracle_calls 100000

    # (GUASS 18) python3 if_baseline.py --target_pdb_id sample25 --log_if_baseline_robot True --M 10 --tau 5 
    # (GUASS 18) python3 if_baseline.py --target_pdb_id sample25 --log_if_baseline_robot True --M 5 --tau 10
    # (GUASS 4) python3 if_baseline.py --target_pdb_id sample25 --log_if_baseline_robot True --M 5 --tau 20
    # (GUASS 18) python3 if_baseline.py --target_pdb_id sample25 --log_if_baseline_robot True --M 20 --tau 5
    # ALL COMBOS ARE WINS! 

    # CUDA_VISIBLE_DEVICES=1 python3 if_baseline.py --target_pdb_id sample199 --log_if_baseline_robot True --M 10 --tau 5
    # CUDA_VISIBLE_DEVICES=2 python3 if_baseline.py --target_pdb_id sample199 --log_if_baseline_robot True --M 5 --tau 10
    # CUDA_VISIBLE_DEVICES=5 python3 if_baseline.py --target_pdb_id sample199 --log_if_baseline_robot True --M 5 --tau 20
    # CUDA_VISIBLE_DEVICES=5 python3 if_baseline.py --target_pdb_id sample199 --log_if_baseline_robot True --M 20 --tau 5

    # python3 if_baseline.py --target_pdb_id sample25 --compute_probs_h True  (done)
    # python3 if_baseline.py --target_pdb_id sample25 --log_if_baseline_constrained True --min_prob_human 0.8  (gauss17)
    # python3 if_baseline.py --target_pdb_id sample25 --log_if_baseline_constrained True --min_prob_human 0.9  (gauss18)

    # python3 if_baseline.py --target_pdb_id sample199 --compute_probs_h True  (done)
    # TODO: 
    # python3 if_baseline.py --target_pdb_id sample199 --log_if_baseline_constrained True --min_prob_human 0.8 (ON gauss19)
    # python3 if_baseline.py --target_pdb_id sample199 --log_if_baseline_constrained True --min_prob_human 0.9  (ON gauss3)

    # python3 if_baseline.py --target_pdb_id sample228 --min_prob_human -1 (running on gauss)

    if args.compute_probs_h:
        compute_and_save_if_baseline_human_probs(the_target_pdb_id=args.target_pdb_id)
    elif args.analyze_probs_human:
        analyze_probs_human() 
    elif args.log_if_baseline_constrained:
        log_if_baseline_constrained(
            target_pdb_id=args.target_pdb_id, 
            min_prob_human=args.min_prob_human,
        )
    elif args.log_if_baseline_robot:
        log_if_baseline_robot(
            target_pdb_id=args.target_pdb_id, 
            M=args.M,
            tau=args.tau,
            step_size=args.step_size,
        )
    elif args.all_robot:
        target_pdb_id_nums = [587,359,280,337,459,582,615,1104] # 286, 199, 25 done 
        ms = [5, 10, 20]
        taus = [5, 10, 20, 50, 100]
        for target_id_num in target_pdb_id_nums:
            for m__ in ms:
                for tau__ in taus:
                    log_if_baseline_robot(
                        target_pdb_id=f"sample{target_id_num}", 
                        M=m__,
                        tau=tau__,
                        step_size=args.step_size,
                    )
    else:
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
