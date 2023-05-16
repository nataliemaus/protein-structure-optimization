# protein-structure-optimization
Optimize AA seqs with desired folded 3D structure 


# Allow permission: 
cd oracle/
chmod 701 TMalign

# Locust/ 6000
docker run -v /home1/n/nmaus/protein-structure-optimization/:/workspace/protein-structure-optimization --gpus all -it nmaus/fold2

# Allegro: 
docker pull docker.io/nmaus/fold2:latest
docker run -v /home/nmaus/protein-structure-optimization/:/workspace/protein-structure-optimization -e NVIDIA_VISIBLE_DEVICES=6 -it nmaus/fold2:latest 

# OTHER
docker run -v /home/nmaus/protein-structure-optimization/:/workspace/protein-structure-optimization --gpus all -it nmaus/fold2 

docker run -v /home/nmaus/protein-structure-optimization/:/workspace/protein-structure-optimization --gpus "device=1" -it nmaus/fold2 

# VIVANCE
docker run --privileged --gpus all -v /home/nmaus/protein-structure-optimization/:/workspace/protein-structure-optimization -w /workspace/protein-structure-optimization/lolbo_scripts -it nmaus/fold2:latest 

# PRESTO
docker run -v /home/nmaus/protein-structure-optimization/:/workspace/protein-structure-optimization -w /workspace/protein-structure-optimization/lolbo_scripts --gpus "device=0" -it nmaus/fold2 

docker run -v /home/nmaus/protein-structure-optimization/:/workspace/protein-structure-optimization -w /workspace/protein-structure-optimization/lolbo_scripts --gpus "device=0" -d nmaus/fold2 COMMAND

docker run -v /home/nmaus/protein-structure-optimization/:/workspace/protein-structure-optimization -w /workspace/protein-structure-optimization/lolbo_scripts --gpus "device=5" -d nmaus/fold2:latest python3 ... 

# Gauss 
runai delete job lolbo-opt19
runai submit lolbo-opt11 -v /shared_data0/protein-structure-optimization/:/workspace/protein-structure-optimization/ --working-dir /workspace/protein-structure-optimization/lolbo_scripts -i nmaus/fold2:latest -g 1 --interactive --attach 
runai attach lolbo-opt0

# EC2 
docker run --privileged --gpus all -v /home/ec2-user/protein-structure-optimization/:/workspace/protein-structure-optimization -w /workspace/protein-structure-optimization/lolbo_scripts -it nmaus/fold2:latest 


# JKGARDNER XXX no work!!! 
docker run -v /home/nmaus/protein-structure-optimization/:/workspace/protein-structure-optimization --gpus all -it nmaus/fold2 

# IF BASELINE RUN: 
cd lolbo_scripts 

CUDA_VISIBLE_DEVICES=3 python3 if_baseline.py --target_pdb_id 2l67 

# SAVE DATA:
CUDA_VISIBLE_DEVICES=4 python3 create_initialization_data.py --num_seqs 1000 --bsz 10 --target_pdb_id sample228 

runai attach lolbo-opt2 

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
runai delete job lolbo-opt8 
runai submit lolbo-opt8 -v /shared_data0/protein-structure-optimization/:/workspace/protein-structure-optimization/ --working-dir /workspace/protein-structure-optimization/lolbo_scripts -i nmaus/fold2 -g 1 --interactive --attach 
runai attach lolbo-opt8


runai attach test1

runai delete job test1

# IF BASELINE! ... 
CUDA_VISIBLE_DEVICES=0 
python3 if_baseline.py --target_pdb_id sample228 --save_freq 100000

python3 if_baseline.py --compute_probs_h True 

# docker run --privileged --gpus all -it nmaus/fold2:latest



________________________________________
________________________________________
________________________________________
________________________________________
________________________________________
________________________________________

# LOLBO OPTIMIZE TM... : 

cd lolbo_scripts 

CUDA_VISIBLE_DEVICES=0 

docker run -v /home/nmaus/protein-structure-optimization/:/workspace/protein-structure-optimization -w /workspace/protein-structure-optimization/lolbo_scripts --gpus "device=2" -d nmaus/fold2:latest 

CUDA_VISIBLE_DEVICES=2 

python3 tm_optimization.py --task_id tm --track_with_wandb True --wandb_entity nmaus --num_initialization_points 15000 --max_n_oracle_calls 150000 --bsz 10 --dim 1024 --max_string_length 150 --vae_tokens uniref --init_w_esmif True --target_pdb_id sample1104 --min_prob_human 0.8 - run_lolbo - done 

# constrained: --min_prob_human 0.8 

YIMENG SET w/ NEW UNIREF VAE MODEL (esm if init only!)
- == done, above hline == averaged over many 
_________________constrained human 0.8 15k init_______________________________
CODE UP SO WE DON'T RECOMPUTE THOSE 15K EVERY TIME!! 
199 EC221 -
455 GAUSS17 GAUSS0
582 GAUSS18
615 ALLEGRO6 ALLEGRO7
587 PRESTO2 GAUSS1
286 EC210 GAUSS2
25 EC222 - 
1104 EC211 GAUSS11(naan death, restarting w/ 100k init...)
280 EC212 
337 EC220
459 EC223

_________________constrained human 0.8_______________________________
286 :( 10k would be huge 
25 - 
199 - 
587 EC221-0.8-X1    10k would be huge 
280 EC222-0.8-X1     close, 10k would be huge 
337 GAUSS10-0.8-X1   close, 10k would be huge 
459 GAUSS16-0.8-X1   15k would be huge 
582 ALLEGRO5-0.8-X1  15k would be huge 
615 ALLEGRO1-0.8-X2  10 huge, 12k better 
1104 ALLEGRO2,3-0.8-X2  init won't help, just run more 
455 ALLEGRO4-0.8-X1   5k would be huge! 

__________ROBOT__________________________
# ROBOT: 
docker run -v /home/nmaus/protein-structure-optimization/:/workspace/protein-structure-optimization -w /workspace/protein-structure-optimization/robot_scripts --gpus "device=5" -d nmaus/fold2:latest 

CUDA_VISIBLE_DEVICES=4 

python3 diverse_tm_optimization.py --task_id tm --max_n_oracle_calls 150000 --bsz 10 --save_csv_frequency 10 --track_with_wandb True --wandb_entity nmaus --num_initialization_points 1000 --dim 1024 --vae_tokens uniref --max_string_length 150 --init_w_esmif True --M 5 --tau 20 --target_pdb_id sample459 - run_robot - done 

25 m10t5-X3 
25 m20t5 - 
199 m10t5-X3 
199 m20t5-X2 
199 m5t10-X2 
199 m5t20-X0 - GAUSS19 GAUSS3 W already 
582 m10t5 PRESTO4 
582 m5t10 PRESTO5 
455 m10t5 VIVANCE7 
455 m5t10 PRESTO0  
1104 m10t5 PRESTO1 
1104 m5t10 PRESTO3 

587 m5t20 LOCUST0 GAUSS4 GAUSS5
280 m5t20 LOCUST1 GAUSS6 GAUSS7
582 m5t20 VIVANCE3 LOCUST2 GAUSS8 
455 m5t20 VIVANCE6 LOCUST4 GAUSS12
1104 m5t20 VIVANCE5  WIN
286 m5t20 VIVANCE0 LOCUST5 GAUSS13
337 m5t20 VIVANCE1 LOCUST6 GAUSS14 
459 m5t20 VIVANCE2 LOCUST7 GAUSS15 
615 m5t20 VIVANCE4 ALLEGRO0 LOCUST3












____________________avareged____________________________
587 -- *
359 -- XXX MESSED UP SAVING SOMEHOW ?? (COULD LOOK INTO)
228 -- XXX BASELINE NOT SAVED XXX 
199 -- *
25 -- *
________________________________________________
280 - *
337 - *
459 - *
582 - *
615 - *
455 -  * 
1104 - * 
286 -- *
494 -       
129 -
611 - 
65 - 
583 -
363 - 
458 -  
479 - 
215 -
664 - 
117 EC2-12 (W, just let finish)
375 :(
3106 - :(  
575 - :( 
167 * :(
41 - :( 
668 - :(

# CUDA_VISIBLE_DEVICES=0 (DO NOT KILL MORE! NEED SAVED DATA!)
# python3 if_baseline.py --target_pdb_id sample167  
____ ??? 
NEW 10: (breifly started first 6 below and killed )
135 ALLEGRO
374 VIVANCE 
527 VIVANCE 
213 GAUSS
569 GAUSS
386 GAUSS
437 
499 
254 
651 

python3 if_baseline.py --target_pdb_id sample386 --max_n_oracle_calls 100000
 


# total: 30 (all running for both baseline + regular)







# --gvp_vae True --vae_kl_factor 0.001 --dim 1536 --update_e2e False   XXX never again XXX 
# constrained: --min_prob_human 0.8   
#       (Note: no constraints on locust (too little storage...))
#       only constrs on ones where baseline will finish (need saved data)



_______________________________________

Could Add: 
IF DOES VERY GOOD (0.95):
350 
253   
537 
174 
647
292
486
478
295
591
216
126
IF DOES VERY MEDIUM (0.7-0.8)

DO BAD: 
424
585 
616
579
4107 



_______________________________________
_______________________________________




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

docker run -v /home/nmaus/protein-structure-optimization/:/workspace/protein-structure-optimization -w /workspace/protein-structure-optimization/lolbo_scripts --gpus "device=5" -d nmaus/fold2 python3 ... 
if_baseline.py --target_pdb_id sample286


## RUNAI GAUSS 
runai submit robot-struct1 -v /shared_data0/protein-structure-optimization/:/workspace/protein-structure-optimization/ --working-dir /workspace/antibody-design/robot_scripts -i nmaus/fold2 -g 1 \ --command -- python3 diverse_tm_optimization.py ... 



