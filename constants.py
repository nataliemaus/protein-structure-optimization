''' Constants used by optimization routine '''


DEBUG_MODE=False 
ALL_AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


# dim512_k1_kl0001_acc94_vivid-cherry-17_model_state_newest.pkl     
# dim512_k1_kl001_acc89_radiant-sun-31_epoch_35.pkl    


# UNIREF VAE 
VAE_DIM_TO_STATE_DICT_PATH = {}
# seperate by type of tokens used to train (uniref 3-mers or esm token
VAE_DIM_TO_STATE_DICT_PATH["esm"] = {} 
VAE_DIM_TO_STATE_DICT_PATH["uniref"] = {} 
# VAE_DIM_TO_STATE_DICT_PATH["uniref"][1024] = "../uniref_vae/saved_models/fiery-plasma-33_model_state.pkl" 
# best regular uniref model: 
VAE_DIM_TO_STATE_DICT_PATH["uniref"][1024] = "../uniref_vae/saved_models/dim512_k1_kl0001_acc94_vivid-cherry-17_model_state_newest.pkl"


VAE_DIM_TO_STATE_DICT_PATH["esm"][1024] = "../uniref_vae/saved_models/cerulean-bee-51_model_state.pkl" 
VAE_DIM_TO_STATE_DICT_PATH["esm"][512] = "../uniref_vae/saved_models/iconic-star-52_model_state.pkl" 
# VAE_DIM_TO_STATE_DICT_PATH["esm"][1024] = "../uniref_vae/saved_models/cerulean-spaceship-32_epoch_20.pkl" 



# cerulean-spaceship-32_epoch_20.pkl 





