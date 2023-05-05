
import os 

def save(
    target_id = "sample359",
    tm_score_ours = 0.8333,
    best_seq = "AAA",
    tm_score_esmif = 0.8068,
    best_if_seq = "AAA",
    fold_model=None,
):
    from oracle.fold import seq_to_pdb
    if not os.path.exists("optimal_pdbs_found/"):
        os.mkdir("optimal_pdbs_found/")

    # save_path = "./output.pdb" 
    save_path = f"optimal_pdbs_found/{target_id}_{tm_score_ours}.pdb"
    seq_to_pdb(best_seq, save_path=save_path, model=fold_model)

    save_path = f"optimal_pdbs_found/{target_id}_{tm_score_esmif}_esmif.pdb"
    seq_to_pdb(best_if_seq, save_path=save_path, model=fold_model)


def align(
    predicted_pdb_path,
    target_pdb_path,
): 
    # ** MUST RELOAD EACH TIME OR BREAKS! 
    import pymolPy3
    pm = pymolPy3.pymolPy3(0)
    save_aligned_pdb_path = predicted_pdb_path.replace(".pdb", "_aligned.pdb")
    predicted_name = predicted_pdb_path.split("/")[-1].split(".")[-2]
    target_name = target_pdb_path.split("/")[-1].split(".")[-2]
    pm('delete all') 
    pm(f"load {target_pdb_path}")
    pm(f'load {predicted_pdb_path}')
    pm(f'super {predicted_name}, {target_name}')
    pm(f'select old_ab, model {target_name}') # you'll have to figure out which chain is which
    pm(f'remove old_ab') 
    pm(f"save {save_aligned_pdb_path}") 
    pm('delete all') 

def align_optimal_pdbs(
    target_id,
    tm_score_ours,
    tm_score_esmif,
):
    import pymolPy3
    ours_path = f"optimal_pdbs_found/{target_id}_{tm_score_ours}.pdb"
    esmif_path = f"optimal_pdbs_found/{target_id}_{tm_score_esmif}_esmif.pdb"
    target_path = f"oracle/target_pdb_files/{target_id}.pdb"
    align(
        predicted_pdb_path=ours_path,
        target_pdb_path=target_path,
    )
    align(
        predicted_pdb_path=esmif_path,
        target_pdb_path=target_path,
    )




def fold_and_save_main():
    from transformers import EsmForProteinFolding
    fold_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    save(
        target_id = "sample359",
        tm_score_ours = 0.8333,
        best_seq = "MANLMEIKARILILERGLRPETQRREHEAEAEAKNLLAAGKTKLVAAVDARVAELVARGVDFATYLEQALAEDSTRWSKAKCADMMAVARHAHHQRAKAA",
        tm_score_esmif = 0.8068,
        best_if_seq = "HMAMHEIVDLLTELALGHSPLGAGLAQEDEQTGRELLAAGKQSLVAEIRQWVDEIAAEGKDVGAMLADRVRESEKTLSRSTVKLMMSVARAEVRARLAAG",
        fold_model=fold_model,
    )
    save(
        target_id = "sample25",
        tm_score_ours = 0.7846,
        best_seq = "GAAQLQTRLRKEKQEALLKAQNIVDVKVSSEEEETLKTLVEGLNSVGEVKTSSDSEPRNFEKVSVTVIGLMMPNGWVLTYRVRKAIYDKDETVSLSLALRKQ",
        tm_score_esmif = 0.7664,
        best_if_seq = "HHHHHHSSGRKQRRAAEERKSRGFVEIEISPEEEAALKKVEEKLSKKAKCEVQSLQLPDSESVEIRRFTWRNADGTIMVFETSTMKINDTEKKYLRIDLLKE",
        fold_model=fold_model,
    )

def align_main():
    target_id = "sample359"
    tm_score_ours = 0.8333
    tm_score_esmif = 0.8068
    align_optimal_pdbs(
        target_id,
        tm_score_ours,
        tm_score_esmif,
    )

    target_id = "sample25"
    tm_score_ours = 0.7846
    tm_score_esmif = 0.7664
    align_optimal_pdbs(
        target_id,
        tm_score_ours,
        tm_score_esmif,
    )


if __name__ == "__main__":
    print("main")
    align_main() 