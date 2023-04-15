# protein-structure-optimization
Optimize AA seqs with desired folded 3D structure 


# Allow permission: 
cd oracle/
chmod 701 TMalign

# Locust/ 6000
docker run -v /home1/n/nmaus/protein-structure-optimization/:/workspace/protein-structure-optimization --gpus all -it nmaus/fold2

# IF BASELINE RUN: 
cd lolbo_scripts 

CUDA_VISIBLE_DEVICES=7 python3 if_baseline.py --target_pdb_id 2lwe

# SAVE DATA:
CUDA_VISIBLE_DEVICES=7 python3 create_initialization_data.py --num_seqs 10000 --bsz 10 --target_pdb_id 300_28

runai attach lolbo-opt2 

# new harder ones 
# 17044
# 24016
# 2609 
# 2702 
# 2703 
# 27014 
# 30016 
# 30028 

# saving 10k on locust for:(lowest scorign 6)
4gmq, d
2lwx, d
6qb2, d
6w3d, d
2k3j, d
2lwe,
5njn,
2mn4,
6vg7,
3leq,
7cfv,
6l7k,
2l67,
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

CUDA_VISIBLE_DEVICES=7 python3 tm_optimization.py --task_id tm --track_with_wandb True --wandb_entity nmaus --num_initialization_points 10000 --max_n_oracle_calls 5000000000000 --bsz 2 --max_string_length 102 --dim 1024 --target_pdb_id 30028 - run_lolbo - done 

# new harder ones 
# 17044 X1
# 24016 X1 
# 2609 X1 
# 2702 X1 
# 2703 X1 
# 27014 X1 
# 30016
# 30028 
## First three numbers give actual seq length ... 


# BEST, num init search... 
2lwx, 10k, 1kx3, 100x4, 10     len 32 
6w3d, 10k, 1k, 100x3, 10         len 40 
4gmq, 10k, 1k, 100, 10     len 32
6qb2, 10k, 1k, 100, 10           len 32

# ALL 
4gmq, X3
2lwx, X3
6qb2, X3
6w3d, X3
2k3j, X2 
2lwe, X1
5njn, X1
2mn4, X1
6vg7, X1
3leq, X1
7cfv, X0
6l7k, X0
2l67, X0

# MAX STRING LENGTH == max_string_length*k ! 
# Gauss  tmux attach -t struct0-19 
# 17_bp_sh3 X4 
# 33_bp_sh3 X5 
# 29_bp_sh3 X5 
# 170_h_ob X5  


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



