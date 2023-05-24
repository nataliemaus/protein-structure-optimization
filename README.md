# protein-structure-optimization
Optimize AA seqs with desired folded 3D structure 


# Allow permission: 
cd oracle/
chmod 701 TMalign

# Locust/ a6000
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
runai submit lolbo-opt12 -v /shared_data0/protein-structure-optimization/:/workspace/protein-structure-optimization/ --working-dir /workspace/protein-structure-optimization/lolbo_scripts -i nmaus/fold2:latest -g 1 --interactive --attach 
runai attach lolbo-opt12

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
runai delete job lolbo-opt20
runai submit lolbo-opt20 -v /shared_data0/protein-structure-optimization/:/workspace/protein-structure-optimization/ --working-dir /workspace/protein-structure-optimization/lolbo_scripts -i nmaus/fold2 -g 1 --interactive --attach 
runai attach lolbo-opt20


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
________________________________________
________________________________________

# LOLBO OPTIMIZE TM... : 

cd lolbo_scripts 

CUDA_VISIBLE_DEVICES=0 

docker run -v /home/nmaus/protein-structure-optimization/:/workspace/protein-structure-optimization -w /workspace/protein-structure-optimization/lolbo_scripts --gpus "device=4" -d nmaus/fold2:latest python3 tm_optimization.py --task_id tm --track_with_wandb True --wandb_entity nmaus --num_initialization_points 148000 --max_n_oracle_calls 200000 --bsz 10 --dim 1024 --max_string_length 150 --vae_tokens uniref --init_w_esmif True --target_pdb_id sample374 --min_prob_human 0.8 - run_lolbo - done 

254 PRESTO0 (148k) 
651 PRESTO1 (148k)
611 PRESTO3 (148k) 
374 PRESTO4 (148k) 

# constrained: --min_prob_human 0.8 

YIMENG SET w/ NEW UNIREF VAE MODEL (esm if init only!)
- == done, above hline == averaged over many 

NOTES: do not kill any current constrained or robot runs, things just take time!

_________________New Regular BO 1k init_______________________________
527 -
254 PRESTO5  W
569 ALLEGRO0 tbd early
135 GAUSS3 tbd early
213 GAUSS9 tbd early
437 GAUSS11 tbd early
386 GAUSS12 tbd early
499 GAUSS15 tbd early 

_________________constrained human 0.8_______________________________
25 - 
199 - 
582 - 
280 - W on 148k init
286 - W on 148k init
615 - barely 15k and 148k 
1104 - W on 148k init 

587 PRESTO2 (15k) GAUSS1 (15k) TRASH
337 TRASH
459 TRASH
455 GAUSS17 (15k)  TRASH

_____NEW BASELINE NOT DONE _________ (still early, need time)
664 GAUSS16 (1k)
228 GAUSS18 (1k)
215 GAUSS19 (1k) 
_____NEW BASELINE DONE _________ (still early, need time)
494 VIVANCE0 (10k) VIVANCE7 (148k)
527 VIVANCE1 (10k) GAUSS7 (148k) W (let finish!)
129 GAUSS14 (10k) VIVANCE6 (148k) 
65 VIVANCE2 (10k) VIVANCE4 (148k) 
135 GAUSS8 (148k) 
__new, tuesday 10am start__ 
458 ALLEGRO2 (148k) 
479 ALLEGRO4 (148k) 
569 ALLEGRO5 (148k) 
117 ALLEGRO6 (148k) 
583 ALLEGRO7 (148k) 
664 GAUSS0 (148k) 
228 GAUSS2  (148k) 
215 GAUSS4 (148k) 
386 GAUSS5 (148k) 
213 GAUSS6 (10k)  
437 GAUSS10 (148k) 
499 GAUSS13  (148k) 
359 VIVANCE3 (148k) 
363 VIVANCE5  (148k) 


__________ROBOT__________________________
# ROBOT: 
docker run -v /home/nmaus/protein-structure-optimization/:/workspace/protein-structure-optimization -w /workspace/protein-structure-optimization/robot_scripts --gpus "device=5" -d nmaus/fold2:latest 

CUDA_VISIBLE_DEVICES=7 

CUDA_VISIBLE_DEVICES=4 python3 diverse_tm_optimization.py --task_id tm --max_n_oracle_calls 200000 --bsz 10 --save_csv_frequency 1000 --track_with_wandb True --wandb_entity nmaus --num_initialization_points 10000 --dim 1024 --vae_tokens uniref --max_string_length 150 --init_w_esmif True --M 5 --tau 20 --target_pdb_id sample135 - run_robot - done 

** 10k --> most success!! --> STICK WITH THAT! 

# M 20 TAU 20 RUNNING (assume 10k init): 
582 LOCUST7 (5k init) LOCUST4 (10k init)
__NEW__
494 LOCUST0
527 LOCUST1 
129 a60002
65 a60003
135 a60004


# M 10 TAU 5 RUNNING (assume 10k init): 
199 LOCUST2
1104 LOCUST3
455 LOCUST5 
587 LOCUST6 

# M 20 TAU 20 RESULTS: 
25 - 
199 - 
1104 -
455 - 
280 - barely win 
337 -  
286 - barely win  
587 - 
615 - YAYYY
459 - barely W 
582 :(  
____ NEW _____ 
494
527

# M 10 TAU 5 RESULTS: 
25 - 
199 
1104 
455 
587 
________

582 m5t10 - BIG WIN!!! (can we use though w/ lower constraint?)


___________IF BASSELINESSSS_____________________________
________________________________________
# IF BASELINESSS. 
CUDA_VISIBLE_DEVICES=4 

docker run -v /home/nmaus/protein-structure-optimization/:/workspace/protein-structure-optimization -w /workspace/protein-structure-optimization/lolbo_scripts --gpus "device=3" -d nmaus/fold2:latest 

python3 if_baseline.py --target_pdb_id sample583  
        
________IF BASELINE________________  
    -b == running baseline comp (robot + constr) 
    -d == done running -b and file downloaded to desktop 
    -u == file uploaded to all other GPUs for constr runs 
    fromVIVANCE == -b done but not -b or -u 
    fromOTHER == -b done for constr only (not ROBOT, gotta do robot same machine)
494 -u
129 -u 
65 -u 
664 fromGAUSS
228 fromGAUSS   
215 fromGAUSS
117 fromALLEGRO  
611 fromPRESTO  
359 fromVIVANCE 
363 fromVIVANCE   
458 fromALLEGRO 
479 fromALLEGRO 
583 fromALLEGRO 
--NEW__
135 -u
374 fromPRESTO
527 -u
213 fromGAUSS 
569 fromALLEGRO 
386 fromGAUSS 
437 fromGAUSS
499 fromGAUSS
254 fromPRESTO
651 fromPRESTO


# -b ::
# RUN FOR EACH IF BASELINE AT SAME TIME: (robot takes no GPU!! )
#   (does robot all ms and taus + probsh + log constr 0.8)

CUDA_VISIBLE_DEVICES=5

docker run -v /home/nmaus/protein-structure-optimization/:/workspace/protein-structure-optimization -w /workspace/protein-structure-optimization/lolbo_scripts --gpus "device=4" -d nmaus/fold2:latest python3 if_baseline.py --target_pdb_id sample374 --compute_and_log_probs_h True 

tmux new -s struct6b 

python3 if_baseline.py --target_pdb_id sample664 --all_robot_one_id True 

    Note: (unforunatley ROBOT one can't run on gauss (requres no GPU), need move all data to viance or somewhere )


# SFTP 
get if_baseline_probs_human_sample135.csv 








________________________________________
________________________________________
________________________________________
________________________________________
________________________________________
________________________________________
________________________________________
________________________________________
________________________________________
________________________________________
________________________________________
________________________________________
________________________________________
________________________________________
________________________________________
________________________________________
________________________________________
________________________________________
________________________________________
________________________________________






_______past_________
25 m10t5-X3 
199 m10t5-X3 
199 m20t5-X2 
199 m5t10-X2 



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
117 - 
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
135 
374 
527
213 
569
386
437 
499 
254 
651 

python3 if_baseline.py --target_pdb_id sample135 --max_n_oracle_calls 100000
 


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



