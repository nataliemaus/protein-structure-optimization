import sys 
sys.path.append("../")
from oracle.fold import inverse_fold_many_seqs
import wandb 
import os
os.environ["WANDB_SILENT"] = "True"


def create_wandb_tracker(
    config_dict,
    wandb_project_name,
    wandb_entity="nmaus",
):
    tracker = wandb.init(
        project=wandb_project_name,
        entity=wandb_entity,
        config=config_dict,
    ) 
    return tracker 

    
# self.wandb_project_name = f"optimimze-{self.task_id}"


inverse_fold_many_seqs(
    target_pdb_id, 
    num_seqs, 
    chain_id="A", 
    model=None
)