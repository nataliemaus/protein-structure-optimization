''' Constants used by optimization routine '''


DEBUG_MODE=False 
ALL_AMINO_ACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

# UNIREF VAE 
VAE_DIM_TO_STATE_DICT_PATH = {}
# seperate by type of tokens used to train (uniref 3-mers or esm tokens) 
VAE_DIM_TO_STATE_DICT_PATH["esm"] = {} 
VAE_DIM_TO_STATE_DICT_PATH["uniref"] = {} 
VAE_DIM_TO_STATE_DICT_PATH["uniref"][1024] = "../uniref_vae/saved_models/fiery-plasma-33_model_state.pkl" 
VAE_DIM_TO_STATE_DICT_PATH["esm"][1024] = "../uniref_vae/saved_models/cerulean-bee-51_model_state.pkl" 
VAE_DIM_TO_STATE_DICT_PATH["esm"][512] = "../uniref_vae/saved_models/iconic-star-52_model_state.pkl" 
# VAE_DIM_TO_STATE_DICT_PATH["esm"][1024] = "../uniref_vae/saved_models/cerulean-spaceship-32_epoch_20.pkl" 



# cerulean-spaceship-32_epoch_20.pkl 





