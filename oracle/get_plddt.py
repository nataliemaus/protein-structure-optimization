from transformers import AutoTokenizer
import torch 
device = "cuda:0"

def compute_plddt(seq, fold_model):
    # Compute plddt score for sequence 
    fold_model = fold_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

    # if seq is not a list of sequences, convert it to a list
    if not isinstance(seq, list):
        seq = [seq]

    tokenized_input = tokenizer(seq, return_tensors="pt", add_special_tokens=False)['input_ids'].to(device)

    with torch.no_grad():
        output = fold_model(tokenized_input)
        # Calculate the mean plddt score for single chain proteins
        mean_plddt = (output["plddt"] * output["atom37_atom_exists"]).sum(
                dim=(1, 2)
            ) / output["atom37_atom_exists"].sum(dim=(1, 2))

    return mean_plddt.item()