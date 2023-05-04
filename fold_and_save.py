
from oracle.fold import seq_to_pdb
import os 
from transformers import EsmForProteinFolding

def save(
    target_id = "sample359",
    tm_score_ours = 0.8333,
    best_seq = "AAA",
    tm_score_esmif = 0.8068,
    best_if_seq = "AAA",
    fold_model=None,
):
    if not os.path.exists("optimal_pdbs_found/"):
        os.mkdir("optimal_pdbs_found/")

    # save_path = "./output.pdb" 
    save_path = f"optimal_pdbs_found/{target_id}_{tm_score_ours}.pdb"
    seq_to_pdb(best_seq, save_path=save_path, model=fold_model)

    save_path = f"optimal_pdbs_found/{target_id}_{tm_score_esmif}_esmif.pdb"
    seq_to_pdb(best_if_seq, save_path=save_path, model=fold_model)


if __name__ == "__main__":
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
        target_id = "sample359",
        tm_score_ours = 0.7846,
        best_seq = "GAAQLQTRLRKEKQEALLKAQNIVDVKVSSEEEETLKTLVEGLNSVGEVKTSSDSEPRNFEKVSVTVIGLMMPNGWVLTYRVRKAIYDKDETVSLSLALRKQ",
        tm_score_esmif = 0.7664,
        best_if_seq = "HHHHHHSSGRKQRRAAEERKSRGFVEIEISPEEEAALKKVEEKLSKKAKCEVQSLQLPDSESVEIRRFTWRNADGTIMVFETSTMKINDTEKKYLRIDLLKE",
        fold_model=fold_model,
    )


