# protein-structure-optimization
Optimize AA seqs with desired folded 3D structure 


# Allow permission: 
cd oracle/
chmod 701 TMalign

# Locust/ 6000
docker run -v /home1/n/nmaus/protein-structure-optimization/:/workspace/protein-structure-optimization --gpus all -it nmaus/fold2

# Allegro: 
docker run -v /home/nmaus/protein-structure-optimization/:/workspace/protein-structure-optimization -e NVIDIA_VISIBLE_DEVICES=1 -it nmaus/fold2 

# OTHER
docker run -v /home/nmaus/protein-structure-optimization/:/workspace/protein-structure-optimization --gpus all -it nmaus/fold2 

docker run -v /home/nmaus/protein-structure-optimization/:/workspace/protein-structure-optimization --gpus "device=1" -it nmaus/fold2 

# PRESTO
docker run -v /home/nmaus/protein-structure-optimization/:/workspace/protein-structure-optimization -w /workspace/protein-structure-optimization/lolbo_scripts --gpus "device=0" -it nmaus/fold2 

docker run -v /home/nmaus/protein-structure-optimization/:/workspace/protein-structure-optimization -w /workspace/protein-structure-optimization/lolbo_scripts --gpus "device=0" -d nmaus/fold2 COMMAND




# JKGARDNER XXX no work!!! 
docker run -v /home/nmaus/protein-structure-optimization/:/workspace/protein-structure-optimization --gpus all -it nmaus/fold2 

# IF BASELINE RUN: 
cd lolbo_scripts 

CUDA_VISIBLE_DEVICES=3 python3 if_baseline.py --target_pdb_id 2l67 

# SAVE DATA:
CUDA_VISIBLE_DEVICES=4 python3 create_initialization_data.py --num_seqs 1000 --bsz 10 --target_pdb_id sample228 


IF DOES VERY BAD
---gauss 
337
215
664
668
611
375
65
280
424
121
458
585
41
583
283
286
587
616
579
4107
___ locust 
167
135
3106
374
527
213
569


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
2lwe, d
5njn, X
2mn4, d
6vg7, X
3leq, d
7cfv, X
6l7k, X
2l67, d
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

# IF BASELINE! ... 
CUDA_VISIBLE_DEVICES=0 
python3 if_baseline.py --target_pdb_id sample228


YIMENG SET w/ NEW UNIREF VAE MODEL, IF BASELINE
25 GAUSS if baseline X1 
286 GAUSS if baseline X1
575 GAUSS if baseline X1 
587 GAUSS if baseline X1 
359 LOCUST if baseline X1 
455 LOCUST if baseline X1 
228 LOCUST if baseline X1 

# LOLBO OPTIMIZE TM... : 

cd lolbo_scripts 

CUDA_VISIBLE_DEVICES=4 python3 tm_optimization.py --task_id tm --track_with_wandb True --wandb_entity nmaus --num_initialization_points 1000 --max_n_oracle_calls 5000000000000 --bsz 10 --dim 1024 --max_string_length 150 --vae_tokens uniref --target_pdb_id sample25 --init_w_esmif True --gvp_vae True --vae_kl_factor 0.001 - run_lolbo - done 

YIMENG SET w/ NEW UNIREF VAE MODEL 
25 GAUSS IF0.70865 len34/102 X4     ALLEGROesmifinit X1   GAUSSesmifinit X4 GAUSSesmifinitGVP X0
286 GAUSS IF0.59618 len34/102 X5     ALLEGROesmifinit X1    GAUSSesmifinit X1 
575 GAUSS IF0.82061 len44/132 X5    ALLEGROesmifinit X1     GAUSSesmifinit X1 
587 GAUSS IF0.59744 len35/105 X5    ALLEGROesmifinit X2     GAUSSesmifinit X1 
359 LOCUST IF0.74537 len34/102 X3   ALLEGROesmifinit X1     GAUSSesmifinit X4 
455 LOCUST IF0.67958 len40/120 X3   ALLEGROesmifinit X1     GAUSSesmifinit X1
228 LOCUST IF0.77884 len41/126 X2   ALLEGROesmifinit X1     GAUSSesmifinit X4 

YIMENG SET (1k done)
25 GAUSS IF0.70865 len34/102 uniref-X3 esm-X1
286 GAUSS IF0.59618 len34/102 uniref-X3 esm-X0 
575 GAUSS IF0.82061 len44/132 uniref-X3 esm-X3 
587 GAUSS IF0.59744 len35/105 uniref-X3 esm-X3 
359 LOCUST IF0.74537 len34/102 uniref-X2 esm-X2 
455 LOCUST IF0.67958 len40/120 uniref-X0 esm-X2 
228 LOCUST IF0.77884 len41/126 uniref-X0 esm-X2 

** LENGTH X 3 IF USING ESM TOKENS BC THEY AREN'T 3-MERS!!! 


IF DOES VERY GOOD (0.95)
350 GAUSS X2
253 GAUSS X
494 GAUSS X
129 GAUSS X
537 LOCUST X
174 LOCUST X
582
647
292
486
478
295
591
216
126

IF DOES VERY MEDIUM (0.8)
575 GAUSS X
199 GAUSS X
386 GAUSS X
479 GAUSS X
437 LOCUST X
459 LOCUST X
101

(0.7)
117 GAUSS X
499 GAUSS X
25 GAUSS X
1104 GAUSS X
615 GAUSS X
254 LOCUST X
363 LOCUST X
651 LOCUST X


---gauss 
337
215
664
668
611
375
65
280
424
121
458
585
41
583
283
286 
587
616
579
4107   
___ locust 
167
135
3106
374
527
213
569


# new harder ones 
# 17044 X4 
# 24016 X2   82
# 2609 X2   88
# 2702 X2   92
# 2703 X1   92
# 27014 X0
# 30016 X0
# 30028 X0
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




######### ROBOT: 

# PRESTO: -d to detach and run in background 
docker run -v /home/nmaus/protein-structure-optimization/:/workspace/protein-structure-optimization -w /workspace/protein-structure-optimization/robot_scripts --gpus "device=5" -d nmaus/fold2 python3 diverse_tm_optimization.py --task_id tm --max_n_oracle_calls 5000000000000000000 --bsz 10 --save_csv_frequency 10 --track_with_wandb True --wandb_entity nmaus --num_initialization_points 1000 --dim 1024 --vae_tokens uniref --max_string_length 150 --M 10 --tau 5 --target_pdb_id sample228 - run_robot - done 

docker run -v /home/nmaus/protein-structure-optimization/:/workspace/protein-structure-optimization -w /workspace/protein-structure-optimization/lolbo_scripts --gpus "device=5" -d nmaus/fold2 python3 if_baseline.py --target_pdb_id sample286

YIMENG SET w/ NEW UNIREF VAE MODEL + ROBOT! (presto! )
25 presto X1  if_baselineX0
286 presto X1  if_baselineX1
575 presto X1  if_baselineX1
587 presto X1  if_baselineX1
359 presto X1  if_baselineX1
455 presto X1   if_baselineX1
228 presto X2 (both GPU 5)   if_baselineX1 


## RUNAI GAUSS 
runai submit robot-struct1 -v /shared_data0/protein-structure-optimization/:/workspace/protein-structure-optimization/ --working-dir /workspace/antibody-design/robot_scripts -i nmaus/fold2 -g 1 \ --command -- python3 diverse_tm_optimization.py ... 



