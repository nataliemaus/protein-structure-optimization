import sys 
sys.path.append("../")
from uniref_vae.esm_tokenizer_data import DataModuleESM
from uniref_vae.esm_transformer_vae import InfoTransformerVAE as EsmInfoTransformerVAE
from uniref_vae.transformer_vae_unbounded import InfoTransformerVAE as OgInfoTransformerVAE
from uniref_vae.data import DataModuleKmers
import torch 
from constants import VAE_DIM_TO_STATE_DICT_PATH


def load_uniref_vae(
    path_to_vae_statedict,
    vae_tokens="uniref",
    vae_kmers_k=1,
    d_model=512, # dim//2
    vae_kl_factor=0.0001,
    max_string_length=150,
):
    if vae_tokens == "uniref": # just all uniref tokens
        data_module = DataModuleKmers(
            batch_size=10,
            k=vae_kmers_k,
            load_data=False,
        )
        dataobj = data_module.train
        vae = OgInfoTransformerVAE(
            dataset=dataobj, 
            d_model=d_model,
            kl_factor=vae_kl_factor,
        ) 
    elif vae_tokens == "esm":
        data_module = DataModuleESM(
            batch_size=10,
            load_data=False,
        )
        dataobj = data_module.train
        vae = EsmInfoTransformerVAE(dataset=dataobj, d_model=d_model)
    else:
        assert 0 

    # load in state dict of trained model:
    if path_to_vae_statedict:
        state_dict = torch.load(path_to_vae_statedict) 
        vae.load_state_dict(state_dict, strict=True) 
    vae = vae.cuda()
    vae = vae.eval()

    # set max string length that VAE can generate
    vae.max_string_length = max_string_length

    return vae, dataobj 


if __name__ == "__main__":
    dim = 1024
    vae, dataobj = load_uniref_vae(
        path_to_vae_statedict=VAE_DIM_TO_STATE_DICT_PATH["uniref"][dim],
        vae_tokens="uniref",
        vae_kmers_k=1,
        d_model=dim//2, # dim//2
        vae_kl_factor=0.001,
        max_string_length=150,
    )