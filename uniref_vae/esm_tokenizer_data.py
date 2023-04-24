import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F 
import pandas as pd
import itertools
import os 
import numpy as nps

STOP_TOKEN = "<eos>"

def load_esm_tokens():
    df = pd.read_csv("../uniref_vae/esm_tokens.csv", header=None)
    return df.values.squeeze().tolist() 

class DataModuleESM(pl.LightningDataModule):
    def __init__(self, batch_size, load_data=True): 
        super().__init__() 
        self.batch_size = batch_size 
        vocab = load_esm_tokens() 
        vocab2idx = {token:i for i, token in enumerate(vocab)}
        self.train  = DatasetESM(dataset='train', vocab=vocab, vocab2idx=vocab2idx, load_data=load_data) 
        self.val    = DatasetESM(dataset='val', vocab=vocab, vocab2idx=vocab2idx, load_data=load_data) 
        self.test   = DatasetESM(dataset='test', vocab=vocab, vocab2idx=vocab2idx, load_data=load_data) 
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, pin_memory=True, shuffle=True, collate_fn=collate_fn, num_workers=10)

    def val_dataloader(self):
        return DataLoader(self.val,   batch_size=self.batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn, num_workers=10)
    
    def test_dataloader(self):
        return DataLoader(self.test,   batch_size=self.batch_size, pin_memory=True, shuffle=False, collate_fn=collate_fn, num_workers=10)


class DatasetESM(Dataset): # asssuming train data 
    def __init__(self, dataset='train', data_path=None, k=3, vocab=None, vocab2idx=None, load_data=False):
        self.dataset = dataset
        self.k = k

        if data_path is None: 
            path_to_data = "../uniref_vae/uniref-small.csv"
        assert vocab is not None 

        if load_data:
            df = pd.read_csv(path_to_data )
            train_seqs = df['sequence'].values  # 1_500_000  sequences 
            # SEQUENCE LENGTHS ANALYSIS:  Max = 299, Min = 100, Mean = 183.03 
            regular_data = [] 
            for seq in train_seqs: 
                tokens_list = []
                for token in seq:
                    if token in vocab:
                        tokens_list.append(token)
                    else:
                        tokens_list.append("X")

                # regular_data.append([token for token in seq]) # list of tokens
                regular_data.append(tokens_list)
        
        self.vocab = vocab 

        if vocab2idx is None:
            self.vocab2idx = { v:i for i, v in enumerate(self.vocab) }
        else:
            self.vocab2idx = vocab2idx
        
        self.data = regular_data 
        # self.data = []
        # if load_data:
        #     for seq in regular_data:
        #         token_num = 0
        #         kmer_tokens = []
        #         while token_num < len(seq):
        #             kmer = seq[token_num:token_num+k]
        #             while len(kmer) < k:
        #                 kmer += '-' # padd so we always have length k 
        #             kmer_tokens.append("".join(kmer)) 
        #             token_num += k 
        #         self.data.append(kmer_tokens) 
        
        num_data = len(self.data) 
        ten_percent = int(num_data/10) 
        five_percent = int(num_data/20) 
        if self.dataset == 'train': # 90 %
            self.data = self.data[0:-ten_percent] 
        elif self.dataset == 'val': # 5 %
            self.data = self.data[-ten_percent:-five_percent] 
        elif self.dataset == 'test': # 5 %
            self.data = self.data[-five_percent:] 
        else: 
            raise RuntimeError("dataset must be one of train, val, test")


    def tokenize_sequence(self, list_of_sequences):   
        ''' 
        Input: list of sequences in standard form (ie 'AGYTVRSGCMGA...')
        Output: List of tokenized sequences where each tokenied sequence is a list of kmers
        '''
        tokenized_sequences = []
        for seq in list_of_sequences:
            kmer_tokens = [token for token in seq]
            tokenized_sequences.append(kmer_tokens)

        # for seq in list_of_sequences:
        #     token_num = 0
        #     kmer_tokens = []
        #     while token_num < len(seq):
        #         kmer = seq[token_num:token_num + self.k]
        #         while len(kmer) < self.k:
        #             kmer += '-' # padd so we always have length k  
        #         if type(kmer) == list: kmer = "".join(kmer)
        #         kmer_tokens.append(kmer) 
        #         token_num += self.k 
        #     tokenized_sequences.append(kmer_tokens) 
        return tokenized_sequences 

    def encode(self, tokenized_sequence):
        return torch.tensor([self.vocab2idx[s] for s in [*tokenized_sequence, STOP_TOKEN]])

    def decode(self, tokens):
        '''
        Inpput: Iterable of tokens specifying each kmer in a given protien (ie [3085, 8271, 2701, 2686, ...] )
        Output: decoded protien string (ie GYTVRSGCMGA...)
        '''
        dec = [self.vocab[t] for t in tokens]
        # Chop out start token and everything past (and including) first stop token
        stop = dec.index(STOP_TOKEN) if STOP_TOKEN in dec else None # want first stop token
        protien = dec[0:stop] # cut off stop tokens
        while "<start>" in protien: # start at last start token (I've seen one case where it started w/ 2 start tokens)
            start = (1+dec.index("<start>")) 
            protien = protien[start:] 
        protien = "".join(protien) # combine into single string 
        return protien

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.encode(self.data[idx]) 

    @property
    def vocab_size(self):
        return len(self.vocab)


def collate_fn(data):
    # Length of longest molecule in batch 
    max_size = max([x.shape[-1] for x in data])
    return torch.vstack(
        # Pad with stop token
        [F.pad(x, (0, max_size - x.shape[-1]), value=1) for x in data]
    )

