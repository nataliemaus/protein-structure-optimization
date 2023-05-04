from transformers import AutoTokenizer, EsmForProteinFolding
import torch 
from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
import uuid
import os 
import esm 
from esm.inverse_folding.util import CoordBatchConverter
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
if not torch.cuda.is_available():
    print("NO GPU AVAILABLE")
    assert 0 

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
    '''This function is used to convert a pdb file to a sequence using inverse folding
    Default to chain A, since we only support folding a single sequence at the moment
    '''
    # Crystal: 
    pdb_path = f"../oracle/target_cif_files/{target_pdb_id}.cif" 
    if model is None:
        model, _ = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()
    structure = esm.inverse_folding.util.load_structure(pdb_path, chain_id)
    coords, _ = esm.inverse_folding.util.extract_coords_from_structure(structure)
    # Lower sampling temperature typically results in higher sequence recovery but less diversity
    sampled_seq = model.sample(coords, temperature=1e-6)
    return sampled_seq 


def inverse_fold_many_seqs(target_pdb_id, num_seqs, chain_id="A", model=None):
    '''This function is used to convert a pdb file to a sequence using inverse folding
    Default to chain A, since we only support folding a single sequence at the moment
    returns a list of inverse folded seqs
    '''
    # Crystal: 
    pdb_path = f"../oracle/target_cif_files/{target_pdb_id}.cif" 
    if model is None:
        model, _ = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()
    structure = esm.inverse_folding.util.load_structure(pdb_path, chain_id)
    coords, _ = esm.inverse_folding.util.extract_coords_from_structure(structure)
    # Lower sampling temperature typically results in higher sequence recovery but less diversity
    sampled_seqs = model.sample(coords, temperature=1, num_seqs=num_seqs) 
    return sampled_seqs 

def seq_to_pdb(seq, save_path="./output.pdb", model=None):
    # This function is used to fold a sequence to a pdb file
    # Load the model and tokenizer
    if model is None:
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")

    model = model.cuda() 
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")

    ### Added by Natalie, remove tokens in UNIREF that are unsupported by ESM  
    seq = seq.replace("-", "") 
    seq = seq.replace("U", "") 
    seq = seq.replace("X", "") 
    seq = seq.replace("Z", "") 
    seq = seq.replace("O", "") 
    seq = seq.replace("B", "")

    tokenized_input = tokenizer([seq], return_tensors="pt", add_special_tokens=False)['input_ids'].cuda() 

    with torch.no_grad():
        output = model(tokenized_input)

    output = convert_outputs_to_pdb(output)
    with open(save_path, "w") as f:
        # the convert_outputs_to_pdb function returns a list of pdb files, since we only have one sequence, we only need the first one
        f.write(output[0])
    return

def fold_aa_seq(aa_seq, esm_model=None):
    if esm_model is None:
        esm_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").cuda() 
    if not os.path.exists("temp_pdbs/"):
        os.mkdir("temp_pdbs/")
    folded_pdb_path = f"temp_pdbs/{uuid.uuid1()}.pdb"
    seq_to_pdb(seq=aa_seq, save_path=folded_pdb_path, model=esm_model)
    return folded_pdb_path 

def load_esm_if_model():
    if_model, if_alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    if_model = if_model.eval() 
    if_model = if_model.cuda() 
    return if_model, if_alphabet 

def get_gvp_encoding(pdb_path, chain_id='A', model=None, alphabet=None):
    # This function is used to get the GVP encoding of a sequence
    if model is None:
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()
    model = model.cuda() 

    structure = esm.inverse_folding.util.load_structure(pdb_path, chain_id)

    # Extracting Coordinates from Structure
    coords, native_seq = esm.inverse_folding.util.extract_coords_from_structure(structure)
    coords = torch.tensor(coords) # .cuda() 

    batch_converter = CoordBatchConverter(alphabet) # .cuda() 
    batch = [(coords, None, native_seq)]

    coords, confidence, strs, tokens, padding_mask = batch_converter(batch)
    confidence = confidence.cuda() 

    gvp_out = model.encoder.forward_embedding(coords.cuda(), padding_mask=padding_mask.cuda(), confidence=confidence)[1]['gvp_out']

    # gvp_out.shape   torch.Size([1, 123, 512])
    return gvp_out


def aa_seq_to_gvp_encoding(aa_seq, if_model=None, if_alphabet=None, fold_model=None):
    if (if_model is None) or (if_alphabet is None):
        if_model, if_alphabet = load_esm_if_model()
    if fold_model is None: 
        fold_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").cuda() 
    folded_pdb = fold_aa_seq(aa_seq, esm_model=fold_model)
    encoding = get_gvp_encoding(pdb_path=folded_pdb, model=if_model, alphabet=if_alphabet) 
    return encoding


if __name__ == "__main__":
    aa_seq = "AABBCCDDEEFFGG"
    folded_pdb = fold_aa_seq(aa_seq, esm_model=None) 
    import pdb 
    pdb.set_trace() 



