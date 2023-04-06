from transformers import AutoTokenizer, EsmForProteinFolding
import torch 
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
import uuid
import os 
import esm 
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def convert_outputs_to_pdb(outputs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))
    return pdbs


def inverse_fold(target_pdb_id, chain_id="A", model=None):
    # This function is used to convert a pdb file to a sequence using inverse folding
    # Default to chain A, since we only support folding a single sequence at the moment
    # import torch_geometric
    # import torch_sparse 
    # from torch_geometric.nn import MessagePassing
    pdb_path = f"../oracle/target_pdb_files/{target_pdb_id}.ent" 
    if model is None:
        model, _ = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()

    try:
        structure = esm.inverse_folding.util.load_structure(pdb_path, chain_id)
        coords, _ = esm.inverse_folding.util.extract_coords_from_structure(structure)
    except:
        import pdb 
        pdb.set_trace() 

    # Lower sampling temperature typically results in higher sequence recovery but less diversity
    sampled_seq = model.sample(coords, temperature=1e-6)

    # Calculate the TM-score between the inverse folded sequence and the target sequence
    # First, we need to convert the sampled sequence to a pdb file
    # seq_to_pdb(sampled_seq, "./data/inverse_folded.pdb")
    # tm_score = cal_tm_score("./data/inverse_folded.pdb", pdb_path)
    # Remove the pdb file
    # os.remove("./data/inverse_folded.pdb") 
    # print("The inverse folded sequence is: ", sampled_seq)
    # print("The TM-score between the inverse folded sequence and the target sequence is: ", tm_score)

    return sampled_seq # , tm_score


def seq_to_pdb(seq, save_path="./output.pdb", model=None, device=device):
    # This function is used to fold a sequence to a pdb file
    # Load the model and tokenizer
    if model is None:
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")

    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

    ### Added by Natalie 
    seq = seq.replace("-", "") 
    seq = seq.replace("U", "") 
    seq = seq.replace("X", "") 
    seq = seq.replace("Z", "") 
    seq = seq.replace("O", "") 

    try:
        tokenized_input = tokenizer([seq], return_tensors="pt", add_special_tokens=False)['input_ids'].to(device)
    except:
        print("TOKENIZATION ERROR...")
        import pdb 
        pdb.set_trace() 

    with torch.no_grad():
        output = model(tokenized_input)

    output = convert_outputs_to_pdb(output)
    with open(save_path, "w") as f:
        # the convert_outputs_to_pdb function returns a list of pdb files, since we only have one sequence, we only need the first one
        f.write(output[0])
    return

def fold_aa_seq(aa_seq, esm_model=None):
    if esm_model is None:
        esm_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    if not os.path.exists("temp_pdbs/"):
        os.mkdir("temp_pdbs/")
    folded_pdb_path = f"temp_pdbs/{uuid.uuid1()}.pdb"
    seq_to_pdb(seq=aa_seq, save_path=folded_pdb_path, model=esm_model, device=device)
    return folded_pdb_path 

if __name__ == "__main__":
    aa_seq = "AABBCCDDEEFFGG"
    folded_pdb = fold_aa_seq(aa_seq, esm_model=None) 
    import pdb 
    pdb.set_trace() 



