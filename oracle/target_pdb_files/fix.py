import os 
import glob 

files = glob.glob("*.pdb")
for file in files:
    os.rename(file, file.replace("_", ""))
    