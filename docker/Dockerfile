FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel


RUN apt-get update && apt-get install -y wget libxml2 git

RUN pip install fire 
RUN pip install rdkit-pypi==2022.3.1

# RUN pip install "fair-esm[esmfold]"
RUN pip install git+https://github.com/facebookresearch/esm.git
# OpenFold and its remaining dependency
RUN pip install 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'
RUN pip install 'openfold @ git+https://github.com/aqlaboratory/openfold.git@4b41059694619831a7db195b7e0988fc4ff3a307'

RUN pip install wandb

RUN pip install pytorch_lightning
RUN pip install networkx
RUN pip install matplotlib
RUN pip install einops
RUN apt update

# wandb api key to log runs
## ENV WANDB_API_KEY=xxxxx
WORKDIR /root


RUN apt install -y wget
RUN apt install -y vim
RUN apt install -y unzip

RUN pip install markupsafe==2.0.1

RUN conda install -y pyg -c pyg
RUN conda install -y -c conda-forge biopython
RUN conda install -y -c conda-forge biopandas

RUN pip install --upgrade pandas

RUN pip install transformers

RUN pip install torch-scatter
RUN pip install torch-sparse
RUN pip install torch-cluster
RUN pip install torch-spline-conv
RUN pip install torch-geometric

# RUN pip uninstall fair-esm
# RUN pip install git+https://github.com/facebookresearch/esm.git
# CUDA_VISIBLE_DEVICES=5 python3 if_baseline.py --target_pdb_id sample615

RUN pip install biotite
RUN pip install gpytorch 
RUN pip install botorch
ENV WANDB_API_KEY=dfa956c5bfb9fa492ebf9adede99093772919518
WORKDIR /workspace


# docker build -t yimengzeng/inverse_folding .
# docker push yimengzeng/inverse_folding
# docker pull yimengzeng/inverse_folding

# docker build -t nmaus/fold2 .
# docker push nmaus/fold2 
# docker pull nmaus/fold2
# docker run -v /home/nmaus/protein-structure-optimization:/workspace/protein-structure-optimization --gpus all -it nmaus/fold2

# DOEES NOT WORK ON JKGARDNER DUE TO INCOMPATIBLE DEVICE TYPES! 