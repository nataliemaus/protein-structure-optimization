import torch
from esm_tokenizer_data import DataModuleESM
# from transformer_vae_unbounded import InfoTransformerVAE 
from esm_transformer_vae import InfoTransformerVAE 
import time 
import wandb 
import argparse 
import os 
os.environ["WANDB_SILENT"] = "true" 
if not torch.cuda.is_available():
    print("No GPU Available")
    assert 0 

GVP_READY = False 
if GVP_READY:
    from oracle.fold import (
        fold_aa_seq,
        load_esm_if_model,
        get_gvp_encoding,
    )
    from transformers import EsmForProteinFolding


def start_wandb(args_dict):
    tracker = wandb.init(entity="nmaus", project='UNIREF-VAE', config=args_dict) 
    print('running', wandb.run.name) 
    return tracker 


def train(args_dict):
    print("training") 
    tracker = start_wandb(args_dict) 
    model_save_path = 'saved_models/' + wandb.run.name + '_model_state.pkl'  
    datamodule = DataModuleESM(args_dict["batch_size"], load_data=True ) 

    if args_dict['debug']:
        print("Reducing to num points to debug")
        datamodule.train.data = datamodule.train.data[0: args_dict['num_debug']]
        print("now len data: ", len(datamodule.train.data))
        print('first point:', datamodule.train.data[0]) 
    
    tracker.log({'N train':len(datamodule.train.data)}) 
    model = InfoTransformerVAE(dataset=datamodule.train, d_model=args_dict['d_model'])

    if args_dict['load_ckpt']: 
        state_dict = torch.load(args_dict['load_ckpt']) # load state dict 
        model.load_state_dict(state_dict, strict=True) 
    
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    model = model.cuda()    
    optimizer = torch.optim.Adam([ {'params': model.parameters()} ], lr=args_dict['lr']) 
    lowest_loss = torch.inf 
    for epoch in range(args_dict['max_epochs']):
        start_time = time.time() 
        model = model.train()  
        sum_train_loss = 0.0 
        num_iters = 0
        for data in train_loader:
            optimizer.zero_grad() 
            input = data.cuda() 
            out_dict = model(input) 
            train_dict = {'train_' + k:out_dict[k] for k in out_dict.keys() }
            tracker.log(train_dict) 
            loss = out_dict['loss'] 
            sum_train_loss += loss.item()  
            num_iters += data.shape[0] # 1
            loss.backward()
            optimizer.step() 
        avg_train_loss = sum_train_loss/num_iters
        tracker.log({'time for train epoch':time.time() - start_time,
                    'avg_train_loss_per_epoch':avg_train_loss,
                    'epochs completed':epoch+1 }) 

        if epoch % args_dict['compute_val_freq'] == 0: 
            start_time = time.time() 
            model = model.eval()  
            sum_val_loss = 0.0 
            num_val_iters = 0
            for data in val_loader:
                input = data.cuda() 
                out_dict = model(input)
                sum_val_loss += out_dict['loss'].item() 
                num_val_iters += data.shape[0] # 1 
                val_dict = {'val_' + k:out_dict[k] for k in out_dict.keys() }
                tracker.log(val_dict) 
            tracker.log({'time for val epoch':time.time() - start_time})
            avg_val_loss = sum_val_loss/num_val_iters 
            tracker.log({'avg_val_loss':avg_val_loss, 'epochs completed':epoch+1}) 

            if avg_val_loss < lowest_loss: 
                lowest_loss = avg_val_loss 
                tracker.log({'lowest avg val loss': lowest_loss, 
                                    'saved model at end epoch': epoch+1 }) 
                torch.save(model.state_dict(), model_save_path) 



def run_training():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--debug', type=bool, default=False)  
    parser.add_argument('--flag_esm_tokenizer', type=bool, default=True)  
    parser.add_argument('--use_gvp', type=bool, default=False)  
    parser.add_argument('--lr', type=float, default=0.0001)  
    parser.add_argument('--compute_val_freq', type=int, default=5 )  
    parser.add_argument('--load_ckpt', default="" )  
    parser.add_argument('--max_epochs', type=int, default=100_000 )  
    parser.add_argument('--num_debug', type=int, default=100 )  
    parser.add_argument('--batch_size', type=int, default=128 )  
    parser.add_argument('--d_model', type=int, default=128 )
    args = parser.parse_args() 
    args_dict = vars(args)
    train(args_dict) 


def test_gvp():
    if_model, if_alphabet = load_esm_if_model()
    aa_seq = 'MEELLKKILEEVKKLEEELKKLEGLEPELKPLLEKLKEELEKLLEELEKLKEEGKEELPEELLEKLLEELEKLEEELEELLEELEELLEGLEELEELKELFEELKEKLEELKELLEELKEE'
    fold_model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").cuda() 
    folded_pdb = fold_aa_seq(aa_seq, esm_model=fold_model)
    encoding = get_gvp_encoding(pdb_path=folded_pdb, model=if_model, alphabet=if_alphabet) 
    print(encoding.shape)  # torch.Size([1, 123, 512]) = 1, seq_len, 512 
    import pdb 
    pdb.set_trace() 


if __name__ == "__main__":
    print("main")
    run_training()
    # test_gvp()
    # CUDA_VISIBLE_DEVICES=0 python3 train_with_gvp_encoding.py --debug True
    # CUDA_VISIBLE_DEVICES=0 python3 train_with_gvp_encoding.py --d_model 256 

    # conda activate lolbo_mols
    # conda activate bighat2 
    # CUDA_VISIBLE_DEVICES=9 python3 train.py --lr 0.001 --compute_val_freq 50 --k 3 --d_model 128 
    # CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.0005 --d_model 256  
