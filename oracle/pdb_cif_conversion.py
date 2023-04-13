import pymol2

def convert_pdb_to_cif(pdb_file):
    with pymol2.PyMOL() as pymol:
        pymol.cmd.load(pdb_file,'myprotein')
        pymol.cmd.save(pdb_file.replace('.pdb', '.cif'), selection='myprotein')

def convert_ent_to_cif(pdb_file):
    with pymol2.PyMOL() as pymol:
        pymol.cmd.load(pdb_file,'myprotein')
        pymol.cmd.save(pdb_file.replace('.ent', '.cif'), selection='myprotein')

def convert_cif_to_pdb(pdb_file):
    with pymol2.PyMOL() as pymol:
        pymol.cmd.load(pdb_file,'myprotein')
        pymol.cmd.save(pdb_file.replace('.cif', '.pdb'), selection='myprotein')

if __name__ == "__main__":
    # conda activate pymol 
    # 170_44
    # 240_16
    # 260_9
    # 270_2
    # 270_3
    # 270_14 
    # 300_16
    # 300_28 
    pdb_file = "target_pdb_files/300_28.pdb"
    convert_pdb_to_cif(pdb_file)
