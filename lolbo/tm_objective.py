import numpy as np
import torch 
import sys 
sys.path.append("../")
from lolbo.latent_space_objective import LatentSpaceObjective
from constants import (
    DEBUG_MODE,
    VAE_DIM_TO_STATE_DICT_PATH,
    GVP_VAE_STATE_DICT_PATH,
)
from transformers import EsmForProteinFolding
from oracle.aa_seq_to_tm_score import aa_seq_to_tm_score
import os 
from uniref_vae.data import collate_fn
from uniref_vae.load_uniref_vae import load_uniref_vae, load_gvp_vae 
from oracle.fold import load_esm_if_model, aa_seqs_list_to_avg_gvp_encodings, get_gvp_encoding
from oracle.get_prob_human import load_human_classier_model, get_prob_human


class TMObjective(LatentSpaceObjective):
    '''Objective class supports all antibody protein
         optimization tasks and uses the UNIREF VAE by default 
    '''
    def __init__(
        self,
        task_id='tm',
        xs_to_scores_dict={},
        max_string_length=100,
        num_calls=0,
        target_pdb_id="17_bp_sh3",
        dim=1024,
        init_vae=True,
        vae_tokens="uniref",
        vae_kmers_k=1,
        vae_kl_factor=0.0001,
        gvp_vae=False,
        gvp_vae_version_flag=3,
        min_prob_human=-1,
    ):
        self.vae_tokens             = vae_tokens 
        assert vae_tokens in ["esm", "uniref"] 
        self.dim                    = dim # SELFIES VAE DEFAULT LATENT SPACE DIM 
        self.max_string_length      = max_string_length # max string length that VAE can generate
        self.target_pdb_id          = target_pdb_id 
        self.vae_kmers_k            = vae_kmers_k
        self.vae_kl_factor          = vae_kl_factor
        self.gvp_vae                = gvp_vae
        self.gvp_vae_version_flag   = gvp_vae_version_flag
        self.min_prob_human         = min_prob_human 

        if self.min_prob_human != -1:
            self.human_classifier_tokenizer, self.human_classifier_model = load_human_classier_model() 

        try: 
            self.target_pdb_path = f"../oracle/target_pdb_files/{target_pdb_id}.ent"
            assert os.path.exists(self.target_pdb_path)
        except:
            self.target_pdb_path = f"../oracle/target_pdb_files/{target_pdb_id}.pdb"
            assert os.path.exists(self.target_pdb_path)

        self.esm_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        self.esm_model = self.esm_model.eval() 
        self.esm_model = self.esm_model.cuda()

        # V3: 
        if self.gvp_vae:
            self.path_to_vae_statedict = GVP_VAE_STATE_DICT_PATH
        else:
            self.path_to_vae_statedict  = VAE_DIM_TO_STATE_DICT_PATH[vae_tokens][self.dim] # path to trained vae stat dict 

        if self.gvp_vae:
            self.if_model, self.if_alphabet = load_esm_if_model() # v1... 
            # V2: just get GVP embedding for target (that's it)
            # target_gvp_encoding = get_gvp_encoding(self.target_pdb_path, chain_id='A', model=None, alphabet=None)
            # self.avg_target_gvp_encoding = target_gvp_encoding.nanmean(-2)
            # self.avg_target_gvp_encoding = self.avg_target_gvp_encoding.cuda()

        super().__init__(
            num_calls=num_calls,
            xs_to_scores_dict=xs_to_scores_dict,
            task_id=task_id,
            init_vae=init_vae,
        )

    def vae_decode(self, z):
        '''Input
                z: a tensor latent space points 
            Output
                a corresponding list of the decoded input space 
                items output by vae decoder 
        '''
        if type(z) is np.ndarray: 
            z = torch.from_numpy(z).float()
        z = z.cuda()
        self.vae = self.vae.eval()
        self.vae = self.vae.cuda()

        if self.gvp_vae:
            # z =  (bsz, 1536) = (bsz, 1024 + 512)
            try:
                latent_z = z[:,0:1024].reshape(-1, 2, 512)
                avg_gvp_embedding = z[:,1024:] 
                sample = self.vae.sample(n=1, z=latent_z, encodings=avg_gvp_embedding) 
            except:
                import pdb 
                pdb.set_trace() 
        else:
            sample = self.vae.sample(z=z.reshape(-1, 2, self.dim//2))
        # grab decoded aa strings
        decoded_seqs = [self.dataobj.decode(sample[i]) for i in range(sample.size(-2))]

        # decoded_seqs = [dataobj.decode(sample[i]) for i in range(sample.size(-2))]
        # get rid of X's (deletion) 
        temp = [] 
        for seq in decoded_seqs:
            seq = seq.replace("X", "")
            if len(seq) == 0:
                seq = "AAA" # catch empty string case too... 
            temp.append(seq) 
        decoded_seqs = temp

        return decoded_seqs


    def query_oracle(self, x):
        ''' Input: 
                list of items x (list of aa seqs)
            Output:
                method queries the oracle and returns 
                a LIST of corresponding scores which are y,
                or np.nan in the case that x is an invalid input
                for each item in input list
        '''
        scores = []
        for aa_seq in x:
            if DEBUG_MODE:
                score = aa_seq_to_tm_score(
                    aa_seq, 
                    target_pdb_path=self.target_pdb_path,
                    esm_model=self.esm_model,
                ) 
            else:
                try:
                    score = aa_seq_to_tm_score(
                        aa_seq, 
                        target_pdb_path=self.target_pdb_path,
                        esm_model=self.esm_model,
                    )
                except:
                    score = np.nan
            scores.append(score)

        return scores 


    def initialize_vae(self):
        ''' Sets self.vae to the desired pretrained vae and 
            sets self.dataobj to the corresponding data class 
            used to tokenize inputs, etc. '''
        if self.gvp_vae:
            assert self.dim == 1536 # 1024 for z + 512 for gvp embeddingg 
            self.vae, self.dataobj = load_gvp_vae(
                vae_tokens=self.vae_tokens,
                vae_kmers_k=self.vae_kmers_k,
                # d_model=self.dim//2,
                d_model=512,
                vae_kl_factor=self.vae_kl_factor,
                max_string_length=self.max_string_length,
            ) 
        else:
            self.vae, self.dataobj = load_uniref_vae(
                path_to_vae_statedict=self.path_to_vae_statedict,
                vae_tokens=self.vae_tokens,
                vae_kmers_k=self.vae_kmers_k,
                d_model=self.dim//2,
                vae_kl_factor=self.vae_kl_factor,
                max_string_length=self.max_string_length,
            )
        

    def vae_forward(self, xs_batch):
        ''' Input: 
                a list xs 
            Output: 
                z: tensor of resultant latent space codes 
                    obtained by passing the xs through the encoder
                vae_loss: the total loss of a full forward pass
                    of the batch of xs through the vae 
                    (ie reconstruction error)
        '''
        # assumes xs_batch is a batch of smiles strings 
        tokenized_seqs = self.dataobj.tokenize_sequence(xs_batch)
        encoded_seqs = [self.dataobj.encode(seq).unsqueeze(0) for seq in tokenized_seqs]
        X = collate_fn(encoded_seqs)

        if self.gvp_vae:
            # V1/V3 
            with torch.no_grad(): # V1/V3
                avg_gvp_encoding = aa_seqs_list_to_avg_gvp_encodings(
                    aa_seq_list=xs_batch, 
                    if_model=self.if_model, 
                    if_alphabet=self.if_alphabet, 
                    fold_model=self.esm_model,
                ) # torch.Size([bsz, 512])
            dict = self.vae(X.cuda(), avg_gvp_encoding) 

            # V2: 
            # dict = self.vae(X.cuda(), self.avg_target_gvp_encoding.repeat(X.shape[0], 1)) 
        else:
            dict = self.vae(X.cuda())
        # FOR GVP: *** TypeError: forward() missing 1 required positional argument: 'encodings'
        vae_loss, z = dict['loss'], dict['z'] 
        
        if self.gvp_vae:
            z = z.reshape(-1, 1024) # (bsz, 2, 512)
            z = torch.cat((z, avg_gvp_encoding), -1) # (bsz, 1536)
        else:
            z = z.reshape(-1,self.dim)

        return z, vae_loss


    def compute_constraints(self, xs_batch):
        ''' Input: 
                a list xs 
            Output: 
                c: tensor of size (len(xs),n_constraints) of
                    resultant constraint values, or
                    None of problem is unconstrained
                    Note: constraints, must be of form c(x) <= 0!
        '''
        if self.min_prob_human == -1:
            return None 

        c_vals = []
        for x in xs_batch:
            probh = get_prob_human(
                seq=x, 
                human_tokenizer=self.human_classifier_tokenizer, 
                human_model=self.human_classifier_model, 
            )
            c_val = (probh*-1) + self.min_prob_human
            c_vals.append(c_val)
        
        c_vals = torch.tensor(c_vals).float() 
        return c_vals.unsqueeze(-1) 


if __name__ == "__main__":
    # testing molecule objective
    obj1 = TMObjective() 
    print(obj1.num_calls) 
    dict1 = obj1(torch.randn(10,obj1.dim))
    print(dict1['scores'], obj1.num_calls)
    dict1 = obj1(torch.randn(3,obj1.dim))
    print(dict1['scores'], obj1.num_calls)
    dict1 = obj1(torch.randn(1,obj1.dim))
    print(dict1['scores'], obj1.num_calls)
