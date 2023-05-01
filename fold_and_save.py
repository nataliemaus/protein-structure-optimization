

# from lolbo.tm_objective import TMObjective
from oracle.fold import seq_to_pdb


target_id = "sample587"
tm_score = 0.7616
best_seq = "AQRADQAELAAAAQASAAAAAAAAAAAMMMQLQVVVANNINMTTTDDDDQILQVANDAFDDDFGFGAGHLQQSNIIQNIIQALNNNSNDDDMSVLLLLGDALANWGV"


# save_path = "./output.pdb"
save_path = f"optimal_pdbs_found/{target_id}_{tm_score}.pdb"
seq_to_pdb(best_seq, save_path=save_path, model=None)


tm_score = 0.59744
best_if_seq = "GAMAALAAERRAAALAAAAAAALAAAEAARAAAAAALGDAAARRAALAAAAAARLAAELGDAALRAALAAAAAARLAALDAAPAAAAAAIAAAARARLEALLA"
save_path = f"optimal_pdbs_found/{target_id}_{tm_score}_esmif.pdb"
seq_to_pdb(best_if_seq, save_path=save_path, model=None)

