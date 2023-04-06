import sys 
sys.path.append("../")
import os 
from oracle.fold import fold_aa_seq
from oracle.get_tm import cal_tm_score

def aa_seq_to_tm_score(
    aa_seq, 
    target_pdb_path,
    esm_model=None,
):
    folded_pdb_path = fold_aa_seq(aa_seq, esm_model=esm_model)
    score = cal_tm_score(folded_pdb_path, target_pdb_path)
    os.remove(folded_pdb_path)
    return score 
