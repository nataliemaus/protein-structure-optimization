
# CONSTRAINED: 
# On average, optimization reduces structural error by \jake{XX\%} (standard error $\pm$ \jake{0.XX\%}).

import sys 
sys.path.append("../")
from oracle.compute_rmsd import aa_seq_to_rmsd_score
import pandas as pd 
import numpy as np 
import copy 
from transformers import EsmForProteinFolding

ROBOT_FILENAME = "organized_robot_results_for_plotting.csv"
CBO_FILENAME = "human08_optimization_results.csv"

M = 5
tau = 20 

esm_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
esm_model = esm_model.eval() 
esm_model = esm_model.cuda()

def get_avg_perc_decrease(esmif_values, our_values):
    # e = esmif_values.mean() 
    # b = our_values.mean() 
    # return (e - b) / e 
    perc_decs = (esmif_values - our_values) / esmif_values 
    avg_perc_dec = perc_decs.mean() 
    std_err_perc_dec = perc_decs.std() / np.sqrt(len(perc_decs))
    return avg_perc_dec, std_err_perc_dec 


# ROBOT: 
# Taking the average across these mean TM-scores, 
# optimization reduces structural error by
# \jake{XX\%} (standard error $\pm$ \jake{0.XX\%}) 
# and RMSD by \jake{XX\%} (standard error $\pm$ \jake{0.XX\%}) 
# compared to ESM-IF.
def get_robot_perc_dec_tm():
    df = pd.read_csv(ROBOT_FILENAME )
    our_tms = df["mean_tm_score_ours"].values 
    esmif_tms =  df["mean_tm_score_esmif"].values 
    avg_perc_dec_tm, std_err_perc_dec_tm  = get_avg_perc_decrease(1-esmif_tms, 1-our_tms)
    print("Average percent decrease in tm error:", avg_perc_dec_tm)
    print("Standard Error percent decrease in tm error:", std_err_perc_dec_tm)


def get_robot_perc_dec_rmsd():
    df = pd.read_csv(ROBOT_FILENAME.replace(".csv", "_w_RSMD.csv") )
    our_rmsds = df["mean_rmsd_score_ours"].values 
    esmif_rmsds =  df["mean_rmsd_score_esmif"].values 
    avg_perc_dec, std_err_perc_dec  = get_avg_perc_decrease(esmif_rmsds, our_rmsds)
    print("Average percent decrease in RSMD:", avg_perc_dec)
    print("Standard Error percent decrease in RMSD:", std_err_perc_dec)



def get_cbo_perc_dec_tm():
    df = pd.read_csv(CBO_FILENAME )
    our_tms = df["tm_score_ours"].values 
    esmif_tms =  df["tm_score_esmif"].values 
    avg_perc_dec_tm, std_err_perc_dec_tm  = get_avg_perc_decrease(1-esmif_tms, 1-our_tms)
    print("Average percent decrease in tm:", avg_perc_dec_tm)
    print("Standard Error percent decrease in tm:", std_err_perc_dec_tm)


def get_cbo_perc_dec_rmsd():
    df = pd.read_csv(CBO_FILENAME.replace(".csv", "_w_RSMD.csv") )
    our_rmsds = df["rmsd_score_ours"].values 
    esmif_rmsds =  df["rmsd_score_esmif"].values 
    avg_perc_dec, std_err_perc_dec  = get_avg_perc_decrease(esmif_rmsds, our_rmsds)
    print("Average percent decrease in RSMD:", avg_perc_dec)
    print("Standard Error percent decrease in RMSD:", std_err_perc_dec)

def seq_to_rmsd(seq, target_pdb_id):
    target_pdb_path = f"../oracle/target_pdb_files/{target_pdb_id}.pdb"
    rmsd = aa_seq_to_rmsd_score(
        aa_seq=seq, 
        target_pdb_path=target_pdb_path,
        esm_model=esm_model,
    )
    return rmsd 

def compute_rmsds_robot():
    df = pd.read_csv(ROBOT_FILENAME ) 
    new_df = copy.deepcopy(df)
    target_pdb_ids = df["target_pdb_id"].values 
    
    for i in range(M):
        ours_rmsds_i = [
            seq_to_rmsd(seq, target_pdb_ids[k]) for k, seq in enumerate(df[f"diverse_seq{i+1}_ours"].values)
        ]
        esmif_rmsds_i = [
            seq_to_rmsd(seq, target_pdb_ids[k]) for k, seq in enumerate(df[f"diverse_seq{i+1}_esmif"].values)
        ]

        new_df[f"seq{i+1}_rmsd_ours"] = np.array(ours_rmsds_i)
        new_df[f"seq{i+1}_rmsd_esmif"] = np.array(esmif_rmsds_i)
    
    new_df["mean_rmsd_score_ours"] = new_df[[
        f"seq{j+1}_rmsd_ours" for j in range(M)]
    ].mean(axis=1)

    new_df["mean_rmsd_score_esmif"] = new_df[[
        f"seq{j+1}_rmsd_esmif" for j in range(M)]
    ].mean(axis=1)
    new_df.to_csv(ROBOT_FILENAME.replace(".csv", "_w_RSMD.csv"))


def compute_rmsds_cbo(): 
    df = pd.read_csv(CBO_FILENAME ) 
    new_df = copy.deepcopy(df) 
    target_pdb_ids = df["target_pdb_id"].values 
    rmsds_ours = [seq_to_rmsd(seq, target_pdb_ids[k]) for k, seq in enumerate(df["best_seq_ours"].values)]
    new_df["rmsd_score_ours"] = np.array(rmsds_ours)

    rmsds_esmif = [seq_to_rmsd(seq) for seq in df["best_seq_esmif"].values]
    new_df["rmsd_score_esmif"] = np.array(rmsds_esmif)

    new_df.to_csv(CBO_FILENAME.replace(".csv", "_w_RSMD.csv"))


def get_stats():
    print("ROBOT:")
    get_robot_perc_dec_tm()
    compute_rmsds_robot()
    get_robot_perc_dec_rmsd()

    print("\n CBO:")
    get_cbo_perc_dec_tm()
    compute_rmsds_cbo()
    get_cbo_perc_dec_rmsd() 


if __name__ == "__main__":
    get_stats()
