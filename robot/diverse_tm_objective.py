import numpy as np
import torch 
import sys 
sys.path.append("../")
from robot.latent_space_objective import LatentSpaceObjective
from constants import (
    DEBUG_MODE,
    VAE_DIM_TO_STATE_DICT_PATH,
)
from transformers import EsmForProteinFolding
from oracle.aa_seq_to_tm_score import aa_seq_to_tm_score
from oracle.edit_distance import compute_edit_distance
import os  
from uniref_vae.data import collate_fn
from uniref_vae.load_uniref_vae import load_uniref_vae 

class DiverseTMObjective(LatentSpaceObjective):
    '''Objective class supports all antibody IGIH heavy chain
         optimization tasks and uses the OAS VAE by default '''

    def __init__(
        self,
        task_id='tm',
        xs_to_scores_dict={},
        max_string_length=100,
        num_calls=0,
        dim=1024,
        target_pdb_id="17_bp_sh3",
        lb=None,
        ub=None,
        vae_tokens="uniref",
        vae_kmers_k=1,
        vae_kl_factor=0.0001,
    ):
        self.vae_kmers_k            = vae_kmers_k
        self.vae_kl_factor          = vae_kl_factor
        self.vae_tokens             = vae_tokens 
        assert vae_tokens in ["esm", "uniref"] 
        self.dim                    = dim # SELFIES VAE DEFAULT LATENT SPACE DIM
        self.path_to_vae_statedict  = VAE_DIM_TO_STATE_DICT_PATH[self.vae_tokens][self.dim] # path to trained vae stat dict
        self.max_string_length      = max_string_length # max string length that VAE can generate
        self.target_pdb_id          = target_pdb_id 
        try: 
            self.target_pdb_path = f"../oracle/target_pdb_files/{target_pdb_id}.ent"
            assert os.path.exists(self.target_pdb_path)
        except:
            self.target_pdb_path = f"../oracle/target_pdb_files/{target_pdb_id}.pdb"
            assert os.path.exists(self.target_pdb_path)
        self.esm_model              = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        self.esm_model = self.esm_model.eval() 
        self.esm_model = self.esm_model.cuda() 
        
        super().__init__(
            num_calls=num_calls,
            xs_to_scores_dict=xs_to_scores_dict,
            task_id=task_id,
            dim=self.dim, #  DEFAULT VAE LATENT SPACE DIM
            lb=lb,
            ub=ub,
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
        # sample molecular string form VAE decoder
        sample = self.vae.sample(z=z.reshape(-1, 2, self.dim//2))
        # grab decoded aa strings
        decoded_seqs = [self.dataobj.decode(sample[i]) for i in range(sample.size(-2))]

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
                a single input space item x
            Output:
                method queries the oracle and returns 
                the corresponding score y,
                or np.nan in the case that x is an invalid input
        '''
        if len(x) == 0:
            return np.nan 
        if DEBUG_MODE:
            score = aa_seq_to_tm_score(
                x, 
                target_pdb_path=self.target_pdb_path,
                esm_model=self.esm_model,
            )
        else:
            try:
                score = aa_seq_to_tm_score(
                    x, 
                    target_pdb_path=self.target_pdb_path,
                    esm_model=self.esm_model,
                )
            except:
                score = np.nan
        return score  

    def initialize_vae(self):
        ''' Sets self.vae to the desired pretrained vae and 
            sets self.dataobj to the corresponding data class 
            used to tokenize inputs, etc. '''
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
        dict = self.vae(X.cuda())
        vae_loss, z = dict['loss'], dict['z']
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
        return None 

    def divf(self, x1, x2):
        ''' Compute edit distance between two 
            potential optimal sequences so we can
            create a diverse set of optimal solutions
            with some minimum edit distance between eachother'''
        return compute_edit_distance(x1, x2) 

