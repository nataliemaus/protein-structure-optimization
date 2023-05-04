import sys 
sys.path.append("../")
from uniref_vae.esm_tokenizer_data import DataModuleESM
from uniref_vae.esm_transformer_vae import InfoTransformerVAE as EsmInfoTransformerVAE
from uniref_vae.transformer_vae_unbounded import InfoTransformerVAE as OgInfoTransformerVAE
from uniref_vae.data import DataModuleKmers
import torch 
from constants import VAE_DIM_TO_STATE_DICT_PATH, GVP_VAE_STATE_DICT_PATH

from uniref_vae.gvp.gvp_vae_v4 import InfoTransformerVAE as GvpInfoTransformerVAE
from uniref_vae.gvp.data_gvp import DataModuleKmers as GvpDataModuleKmers

def load_gvp_vae(
    vae_tokens="uniref",
    vae_kmers_k=1,
    d_model=512, # dim//2
    vae_kl_factor=0.001,
    max_string_length=150,
):
    assert vae_tokens == "uniref"
    assert vae_kmers_k == 1
    assert vae_kl_factor == 0.001
    assert d_model == 512 
    data_module = GvpDataModuleKmers(
        batch_size=10,
        k=vae_kmers_k,
        load_data=False,
    )
    dataobj = data_module.train

    vae = GvpInfoTransformerVAE(
        dataset=dataobj, 
        d_model=d_model,
        kl_factor=vae_kl_factor,
    ) 

    # load in state dict of trained model:
    state_dict = torch.load(GVP_VAE_STATE_DICT_PATH) 
    vae.load_state_dict(state_dict, strict=True) 
    vae = vae.cuda()
    vae = vae.eval() 

    # set max string length that VAE can generate
    vae.max_string_length = max_string_length

    return vae, dataobj 



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


def test_gvp():
    aa_seq = 'MEELLKKILEEVKKLEEELKKLEGLEPELKPLLEKLKEELEKLLEELEKLKEEGKEELPEELLEKLLEELEKLEEELEELLEELEELLEGLEELEELKELFEELKEKLEELKELLEELKEE'
    aa_seq2 = 'MEELLKKILEEVKKLEEELKKLLLEKLKEELEKLLEELEKLKEEGKEELPEELLEKLLEELEKLEEELEELLLLLEELEELLEGLEELEEL'
    from oracle.fold import aa_seq_to_gvp_encoding, aa_seqs_list_to_avg_gvp_encodings
    from uniref_vae.data import collate_fn 
    aa_seqs_list = [aa_seq, aa_seq2, aa_seq, aa_seq2]


    vae, dataobj = load_gvp_vae() 

    # FOREWARD: 
    tokenized_seqs = dataobj.tokenize_sequence(aa_seqs_list)
    encoded_seqs = [dataobj.encode(seq).unsqueeze(0) for seq in tokenized_seqs]
    X = collate_fn(encoded_seqs)  # torch.Size([1, 122])

    avg_gvp_encodings = aa_seqs_list_to_avg_gvp_encodings(aa_seqs_list, if_model=None, if_alphabet=None, fold_model=None)

    # gvp_encoding = aa_seq_to_gvp_encoding(aa_seq, if_model=None, if_alphabet=None, fold_model=None)
    # print(gvp_encoding.shape) torch.Size([1, 123, 512]) 

    # avg_gvp_encoding = gvp_encodings.nanmean(-2) # torch.Size([1, 512])

    dict = vae(X.cuda(), avg_gvp_encodings) # torch.Size([1, 122])
    vae_loss, z = dict['loss'], dict['z'] 
    # print(z.shape) = torch.Size([1, 2, 512]) 
    dim = 1024 # ?? 
    
    
    # z = z.reshape(-1,dim) # torch.Size([1, 1024])


    # DECODE: 
    # if type(z) is np.ndarray: 
    #     z = torch.from_numpy(z).float()
    z = z.cuda()
    # sample molecular string form VAE decoder
    # sample = vae.sample(1, z=z.reshape(-1, 2, dim//2), encodings=avg_gvp_encodings.cuda() ) 
    sample = vae.sample(1, z=z, encodings=avg_gvp_encodings.cuda() ) 
    decoded_seqs = [dataobj.decode(sample[i]) for i in range(sample.size(-2))]

    import pdb 
    pdb.set_trace() 


if __name__ == "__main__":
    print("main")
    # dim = 1024
    # vae, dataobj = load_uniref_vae(
    #     path_to_vae_statedict=VAE_DIM_TO_STATE_DICT_PATH["uniref"][dim],
    #     vae_tokens="uniref",
    #     vae_kmers_k=1,
    #     d_model=dim//2, # dim//2
    #     vae_kl_factor=0.001,
    #     max_string_length=150,
    # )
    test_gvp() 
     