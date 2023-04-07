# protein-structure-optimization
Optimize AA seqs with desired folded 3D structure 


# Allow permission: 
cd oracle/
chmod 701 TMalign

# Locust/ 6000
docker run -v /home1/n/nmaus/protein-structure-optimization/:/workspace/protein-structure-optimization --gpus all -it nmaus/fold2

# SAVE DATA:
CUDA_VISIBLE_DEVICES=3 python3 create_initialization_data.py --num_seqs 100 --bsz 10 --target_pdb_id 170_h_ob

# all x100 and x10,000 
# 17_bp_sh3
# 33_bp_sh3 
# 29_bp_sh3
# 170_h_ob


runai submit lolbo-struct2 -v /shared_data0/protein-structure-optimization/:/workspace/protein-structure-optimization/ --working-dir /workspace/antibody-design/lolbo_scripts -i nmaus/fold2 -g 1 \ --command -- python3 create_initialization_data.py --num_seqs 20000 --bsz 10 --target_pdb_id 17_bp_sh3 

# running 1000, 20000 

# RUNAI GAUSS INTERACTIVE 
runai submit lolbo-opt19 -v /shared_data0/protein-structure-optimization/:/workspace/protein-structure-optimization/ --working-dir /workspace/protein-structure-optimization/lolbo_scripts -i nmaus/fold2 -g 1 --interactive --attach 

runai attach test1

runai delete job test1

# LOLBO: 

cd lolbo_scripts 

CUDA_VISIBLE_DEVICES=0 python3 tm_optimization.py --task_id tm --track_with_wandb True --wandb_entity nmaus --num_initialization_points 99 --max_n_oracle_calls 500000000 --bsz 10 --max_string_length 60 --dim 1024 --target_pdb_id 17_bp_sh3 - run_lolbo - done 

# Gauss  tmux attach -t struct0-19 
# 17_bp_sh3 X0
# 33_bp_sh3 X0
# 29_bp_sh3 X0
# 170_h_ob X0



## RUNAI GAUSS 
runai submit lolbo-opt-5 -v /shared_data0/protein-structure-optimization/:/workspace/protein-structure-optimization/ --working-dir /workspace/antibody-design/lolbo_scripts -i nmaus/fold2 -g 1 --command -- python3 tm_optimization.py python3 tm_optimization.py --task_id tm --track_with_wandb True --wandb_entity nmaus --num_initialization_points 40 --max_n_oracle_calls 500000000 --bsz 10 --dim 1024 --max_string_length 60 --target_pdb_id 17_bp_sh3 - run_lolbo - done 




# ROBOT: 

CUDA_VISIBLE_DEVICES=0 python3 diverse_tm_optimization.py --task_id tm \
--max_n_oracle_calls 500000000 --bsz 10 --save_csv_frequency 10 \
--track_with_wandb True --wandb_entity nmaus --num_initialization_points 10000 \
--target_pdb_id 17_bp_sh3 --dim 1024 \
--max_string_length 100 --M 10 --tau 5 - run_robot - done 

## RUNAI GAUSS 
runai submit robot-struct1 -v /shared_data0/protein-structure-optimization/:/workspace/protein-structure-optimization/ --working-dir /workspace/antibody-design/robot_scripts -i nmaus/fold2 -g 1 \ --command -- python3 diverse_tm_optimization.py ... 



