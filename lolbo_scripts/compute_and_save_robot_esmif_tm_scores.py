import sys 
sys.path.append("../")
import pandas as pd 
import numpy as np
import random 

DEBUG = False 
if not DEBUG:
    from lolbo.tm_objective import TMObjective

data_path = "../organized_robot_results.csv"

df = pd.read_csv(data_path)

# target_pdb_id,mean_tm_score_ours,mean_tm_score_esmif,diverse_seq1_ours,diverse_seq2_ours,diverse_seq3_ours,diverse_seq4_ours,diverse_seq5_ours,seq1_tm_score_ours,seq2_tm_score_ours,seq3_tm_score_ours,seq4_tm_score_ours,seq5_tm_score_ours,diverse_seq1_esmif,diverse_seq2_esmif,diverse_seq3_esmif,diverse_seq4_esmif,diverse_seq5_esmif
target_pdbs = df["target_pdb_id"]

# if_model, _ = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
#     if_model = if_model.eval().to(device)
# pdb_path = f"../oracle/target_cif_files/{target_pdb_id}.cif"
# structure = esm.inverse_folding.util.load_structure(pdb_path, "A")
#     coords, _ = esm.inverse_folding.util.extract_coords_from_structure(structure)
M = 5


esmif_tm_scores = [] 
for _ in range(M):
    esmif_tm_scores.append([])

for target_pdb_id in df["target_pdb_id"]:
    if not DEBUG:
        objective = TMObjective(
            target_pdb_id=target_pdb_id,
            init_vae=False,
        ) 
    sub_df = df[df["target_pdb_id"]==target_pdb_id]
    sum_tm_scores = 0
    for i in range(M):
        seq = sub_df[f"diverse_seq{i+1}_esmif"].values[0]
        if DEBUG:
            print("seq", seq)
            tm_score = random.randint(0,5)
        else:
            tm_score = objective.query_oracle([seq])[0]
        esmif_tm_scores[i].append(tm_score)
        sum_tm_scores += tm_score
    mean_tm_score = sum_tm_scores/M
    actual_mean = sub_df["mean_tm_score_esmif"].values[0]
    print(f"mean is {mean_tm_score} and should be {actual_mean}")

for i in range(M):
    df[f"seq{i+1}_tm_score_esmif"] = np.array(esmif_tm_scores[i])

df.to_csv("../organized_robot_results_with_esmif_scores.csv", index=None)
# seq1_tm_score_esmifs = []
# seq2_tm_score_esmifs = []
# seq3_tm_score_esmifs = []
# seq4_tm_score_esmifs = []
# seq5_tm_score_esmifs = []



