
from oracle.fold import seq_to_pdb
import os 

def save(
    target_id = "sample359",
    tm_score_ours = 0.8333,
    best_seq = "AAA",
    tm_score_esmif = 0.8068,
    best_if_seq = "AAA"
):
    if not os.path.exists("optimal_pdbs_found/"):
        os.mkdir("optimal_pdbs_found/")

    # save_path = "./output.pdb" 
    save_path = f"optimal_pdbs_found/{target_id}_{tm_score_ours}.pdb"
    seq_to_pdb(best_seq, save_path=save_path, model=None)

    save_path = f"optimal_pdbs_found/{target_id}_{tm_score_esmif}_esmif.pdb"
    seq_to_pdb(best_if_seq, save_path=save_path, model=None)


if __name__ == "__main__":
    save(
        target_id = "sample359",
        tm_score_ours = 0.8333,
        best_seq = "MANLMEIKARILILERGLRPETQRREHEAEAEAKNLLAAGKTKLVAAVDARVAELVARGVDFATYLEQALAEDSTRWSKAKCADMMAVARHAHHQRAKAA",
        tm_score_esmif = 0.8068,
        best_if_seq = "HMAMHEIVDLLTELALGHSPLGAGLAQEDEQTGRELLAAGKQSLVAEIRQWVDEIAAEGKDVGAMLADRVRESEKTLSRSTVKLMMSVARAEVRARLAAG"
    )
    save(
        target_id = "sample359",
        tm_score_ours = 0.7846,
        best_seq = "GAAQLQTRLRKEKQEALLKAQNIVDVKVSSEEEETLKTLVEGLNSVGEVKTSSDSEPRNFEKVSVTVIGLMMPNGWVLTYRVRKAIYDKDETVSLSLALRKQ",
        tm_score_esmif = 0.7664,
        best_if_seq = "HHHHHHSSGRKQRRAAEERKSRGFVEIEISPEEEAALKKVEEKLSKKAKCEVQSLQLPDSESVEIRRFTWRNADGTIMVFETSTMKINDTEKKYLRIDLLKE"
    )


