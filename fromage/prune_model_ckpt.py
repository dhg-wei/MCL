"""Prune pretrained model weights to reduce size.

This keeps only the weights that we finetune, and discards the pretrained LLM / visual encoder weights.
"""
import torch
import json
import glob
import os
from collections import OrderedDict
import time

iddx=0

if __name__ == '__main__':
  while True:
    if iddx>0:
        time.sleep(5*60)
    iddx+=1
    # run_path = '/home/users/nus/t0931144/scratch/icml_runs/'
    run_path = '../runs/'
    ckpts = glob.glob(run_path+'*/*ckpt_epoch_*')
    if len(ckpts)>0:
      for ckpt_path in ckpts:
        pruned_output_path = ckpt_path.replace('ckpt_epoch','pretrained_ckpt')
        model_args_path = os.path.dirname(ckpt_path)+ '/model_args.json'
        print(model_args_path)

        with open(model_args_path, 'r') as f:
            model_kwargs = json.load(f)
            ret_token_idx = model_kwargs['retrieval_token_idx']

        with open(ckpt_path, 'rb') as f:
            checkpoint = torch.load(f,map_location='cpu')

        stripped_state_dict = {
            k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items() if 
            ('.lm' not in k and '.visual_model' not in k)
        }
        stripped_state_dict = OrderedDict(sorted(stripped_state_dict.items()))

        del checkpoint['epoch']
        print('Best score:', checkpoint['best_score'])
        del checkpoint['optimizer']
        del checkpoint['scheduler']
        for k, v in stripped_state_dict.items():
            stripped_state_dict[k] = v.detach().clone()
        if isinstance(ret_token_idx, list):
        # Prune the pretrained token embeddings and keep just [RET].
          ret_embedding = stripped_state_dict['model.input_embeddings.weight'][ret_token_idx[0]:ret_token_idx[-1]+1, :].detach().clone()
          stripped_state_dict['ret_input_embeddings.weight'] = ret_embedding.cpu()
        else:
          ret_embedding = stripped_state_dict['model.input_embeddings.weight'][ret_token_idx:ret_token_idx+1, :].detach().clone()
          stripped_state_dict['ret_input_embeddings.weight'] = ret_embedding.cpu()
        with open(pruned_output_path, 'wb') as f:
            torch.save({'state_dict': stripped_state_dict}, f)
        os.remove(ckpt_path)
    time.sleep(60*60)
    print('sleep')