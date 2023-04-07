# for new bighat oracles version 
import sys
sys.path.append("../")
import fire
import torch 
import pandas as pd
import math 
from robot_scripts.optimize import Optimize
from robot.diverse_tm_objective import DiverseTMObjective
from constants import (
    VAE_DIM_TO_STATE_DICT_PATH
)
import math
from lolbo_scripts.create_initialization_data import (
    load_init_data,
    # load_uniref_seqs,
    # load_uniref_scores,
)

class DiverseTMOptimization(Optimize):
    """
    Run LOL-ROBOT Optimization for any aa seq Optimization Task 
    using the pretrained Unireff VAE 
    """
    def __init__(
        self,
        dim: int=1024,
        max_string_length: int=100,
        target_pdb_id: str="17_bp_sh3",
        **kwargs,
    ):
        self.dim=dim
        self.path_to_vae_statedict =VAE_DIM_TO_STATE_DICT_PATH[self.dim] 
        self.max_string_length = max_string_length 
        self.target_pdb_id = target_pdb_id 
        super().__init__(**kwargs)

        # add args to method args dict to be logged by wandb
        self.method_args['diverseopt'] = locals()
        del self.method_args['diverseopt']['self']

    def initialize_objective(self):
        # initialize molecule objective
        self.objective = DiverseTMObjective(
            task_id=self.task_id,
            max_string_length=self.max_string_length,
            dim=self.dim,
            target_pdb_id=self.target_pdb_id,
        )

        return self

    def compute_train_zs(
        self,
        bsz=64
    ):
        init_zs = []
        # make sure vae is in eval mode 
        self.objective.vae.eval() 
        n_batches = math.ceil(len(self.init_train_x)/bsz)
        for i in range(n_batches):
            xs_batch = self.init_train_x[i*bsz:(i+1)*bsz] 
            zs, _ = self.objective.vae_forward(xs_batch)
            init_zs.append(zs.detach().cpu())
        init_zs = torch.cat(init_zs, dim=0)
        # now save the zs so we don't have to recompute them in the future:
        state_dict_file_type = self.objective.path_to_vae_statedict.split('.')[-1] # usually .pt or .ckpt
        path_to_init_train_zs = self.objective.path_to_vae_statedict.replace(f".{state_dict_file_type}", '-train-zs.csv')
        zs_arr = init_zs.cpu().detach().numpy()
        pd.DataFrame(zs_arr).to_csv(path_to_init_train_zs, header=None, index=None) 

        return init_zs

    def load_train_data(self):
        ''' Load in or randomly initialize self.num_initialization_points
            total initial data points to kick-off optimization 
            Must define the following:
                self.init_train_x (a list of x's)
                self.init_train_y (a tensor of scores/y's)
                self.init_train_z (a tensor of corresponding latent space points)
            '''
        # train_x_seqs = load_uniref_seqs() 
        # self.init_train_x = train_x_seqs[0:self.num_initialization_points]
        # self.init_train_y = load_uniref_scores(self.target_pdb_id, num_seqs_load=self.num_initialization_points)
        self.init_train_x, self.init_train_y = load_init_data(
            target_pdb_id=self.target_pdb_id, 
            num_seqs_load=self.num_initialization_points
        )
        self.load_train_z() 
        return self 


    def load_train_z(
        self,
    ):
        state_dict_file_type = self.path_to_vae_statedict.split('.')[-1] # usually .pt or .ckpt
        path_to_init_train_zs = self.path_to_vae_statedict.replace(f".{state_dict_file_type}", '-train-zs.csv')
        # if we have a path to pre-computed train zs for vae, load them
        try:
            zs = pd.read_csv(path_to_init_train_zs, header=None).values
            # make sure we have a sufficient number of saved train zs
            assert len(zs) >= self.num_initialization_points
            zs = zs[0:self.num_initialization_points]
            zs = torch.from_numpy(zs).float()
        # otherwisee, set zs to None 
        except: 
            zs = None 
        self.init_train_z = zs 
        return self 


if __name__ == "__main__":
    fire.Fire(DiverseTMOptimization)
