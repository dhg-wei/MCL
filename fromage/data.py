"""Modified from https://github.com/mlfoundations/open_clip"""

from typing import Optional, Tuple

import collections
import logging
import os
import numpy as np
import pandas as pd
import torch
import torchvision.datasets as datasets
from torchvision import transforms as T
from PIL import Image, ImageFont
from torch.utils.data import Dataset
import io
from fromage import utils
import pickle as pkl
from transformers import CLIPProcessor, CLIPModel,CLIPTextModel,CLIPVisionModel
import random

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def get_dataset(args, split: str, tokenizer, precision: str = 'fp32') -> Dataset:
  assert split in ['train', 'val'
    ], 'Expected split to be one of "train" or "val", got {split} instead.'

  dataset_paths = []
  image_data_dirs = []
  train = split == 'train'

  # Default configs for datasets.
  # Folder structure should look like:
  if split == 'train':
    if 'cc3m' in args.dataset:
      dataset_paths.append(os.path.join(args.dataset_dir, 'cc3m_train_icml.pkl'))
      image_data_dirs.append(args.image_dir)
    else:
      raise NotImplementedError
  else:
    raise NotImplementedError

  if len(dataset_paths) > 1:
    print(f'{len(dataset_paths)} datasets requested: {dataset_paths}')
    dataset = torch.utils.data.ConcatDataset([
      CsvDataset(path, image_dir, tokenizer, 'image',
        'caption', args.visual_model, train=train, max_len=args.max_len, precision=args.precision,
        image_size=args.image_size, retrieval_token_idx=args.retrieval_token_idxr,LenRET=args.LenRET)
      for (path, image_dir) in zip(dataset_paths, image_data_dirs)])
  elif len(dataset_paths) == 1:
    dataset = CsvDataset(dataset_paths[0], image_data_dirs[0], tokenizer, 'image',
      'caption', args.visual_model, train=train, max_len=args.max_len, precision=args.precision,
      image_size=args.image_size, retrieval_token_idx=args.retrieval_token_idx,qaprefix = args.qaprefix,LenRET=args.LenRET)
  else:
    raise ValueError(f'There should be at least one valid dataset, got train={args.dataset}, val={args.val_dataset} instead.')
  return dataset


class CsvDataset(Dataset):
  def __init__(self, input_filename, base_image_dir, tokenizer, img_key,
               caption_key, feature_extractor_model: str,
               train: bool = True, max_len: int = 32, sep="\t", precision: str = 'fp32',
               image_size: int = 224, retrieval_token_idx: int = -1,qaprefix=False,LenRET=1):
    logging.debug(f'Loading tsv data from {input_filename}.')
#     df = pd.read_csv(input_filename, sep=sep)
    self.base_image_dir = base_image_dir
    self.LenRET = LenRET
  
    if self.LenRET==1:
      self.RET_Text = '[RET]'
    else:
      self.RET_Text = ''
      for i in range(self.LenRET):
        self.RET_Text+= f'[RET{i}]'
        
    with open(input_filename ,'rb') as f:
      cc3m_train = pkl.load(f)
    random.shuffle(cc3m_train)

      
    self.images = [item['filename']+'.image_byte' for item in cc3m_train]
    self.captions = [item['text'] for item in cc3m_train]
    

    self.qaprefix = qaprefix

    self.new_captions = [self.refine_caption(item['New_Caption']) for item in cc3m_train]
    self.text_conditions = [self.refine_caption(item['Text_Condition']) for item in cc3m_train]
    assert len(self.images) == len(self.captions)

    self.feature_extractor_model = feature_extractor_model
    self.feature_extractor = utils.get_feature_extractor_for_model(
      feature_extractor_model, image_size=image_size, train=False)
    self.image_size = image_size

    self.tokenizer = tokenizer
    self.max_len = max_len
    self.precision = precision
    self.retrieval_token_idx = retrieval_token_idx

    self.font = None
    self.clip_tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",local_files_only=False)
    logging.debug('Done loading data.')
    self.RETtokens = self.tokenizer(self.RET_Text,return_tensors="pt").input_ids[0][1:]

  def __len__(self):
    return len(self.captions)

  def refine_caption(self,caption):
    try:
      caption = caption.strip()
      if caption[-1] in [',','.','?','!']:
        caption = caption[:-1]
        caption = caption.strip()
      caption = caption+'.'
      return caption
    except:
      return ''
  def __getitem__(self, idx):
    while True:
      image_path = os.path.join(self.base_image_dir, str(self.images[idx]))
      caption = str(self.captions[idx]).lower()
      new_caption = str(self.new_captions[idx]).lower()
      text_condition = str(self.text_conditions[idx]).lower()
      if len(new_caption)==0 or len(text_condition)==0:
        idx = np.random.randint(0, len(self)-1)
        continue
      try:
        if new_caption[0]==' ':
          new_caption = new_caption[1:]
        if caption[0]==' ':
          caption = caption[1:]
        if text_condition[0]==' ':
          text_condition = text_condition[1:]

        with open(image_path, 'rb') as f:
          byte_data = f.read()
        img = Image.open(io.BytesIO(byte_data))

        images = utils.get_pixel_values_for_model(self.feature_extractor, img)

        caption =caption+ '\nit is a photo of' + self.RET_Text
        tokenized_data = self.tokenizer(
          caption,
          return_tensors="pt",
          padding='max_length',
          truncation=True,
          max_length=self.max_len)
        tokens = tokenized_data.input_ids[0]

        caption_len = tokenized_data.attention_mask[0].sum()

        ###
        if self.qaprefix:
          text_condition ='Q:'+text_condition +'\nA:it becomes a photo of'+self.RET_Text
          condition_prefix_len = 'Q:'+text_condition +'\nA:'
          text_condition_caption = 'Q:'+text_condition +'\nA:'+ new_caption
        else:
          text_condition = text_condition+ ' it is a photo of ' +self.RET_Text

        tokenized_data_tc = self.tokenizer(
          text_condition,
          return_tensors="pt",
          padding='max_length',
          truncation=True,
          max_length=self.max_len)
        tokens_tc = tokenized_data_tc.input_ids[0]
        tc_len = tokenized_data_tc.attention_mask[0].sum()
        masks_tc = tokenized_data_tc.attention_mask[0]
        for i in range(len(tokens_tc)):
          if tokens_tc[i] in self.RETtokens:
            masks_tc[i]+=1

        
        tokenized_data_cc = self.tokenizer(
          text_condition_caption,
          return_tensors="pt",
          padding='max_length',
          truncation=True,
          max_length=self.max_len)
        tokens_cc = tokenized_data_cc.input_ids[0]
        cc_len = tokenized_data_cc.attention_mask[0].sum()
        
        tokenized_data_prefixc = self.tokenizer(
          condition_prefix_len,
          return_tensors="pt",
          padding='max_length',
          truncation=True,
          max_length=self.max_len)
        pc_len = tokenized_data_prefixc.attention_mask[0].sum()
        
        new_caption_clip_tokens = self.clip_tokenizer(text=new_caption,return_tensors="pt",truncation=True, padding='max_length',max_length = 30).input_ids[0]
        text_condition_clip_tokens = self.clip_tokenizer(text=text_condition,return_tensors="pt",truncation=True, padding='max_length',max_length = 30).input_ids[0]

        
        ###
        decode_caption = self.tokenizer.decode(tokens, skip_special_tokens=False)
        self.font = self.font or ImageFont.load_default()
        cap_img = utils.create_image_of_text(decode_caption.encode('ascii', 'ignore'), width=self.image_size, nrows=2, font=self.font)
        
        if self.LenRET>1:
          if tokens[-1] not in [self.retrieval_token_idx[-1], self.tokenizer.pad_token_id]:
#               tokens[-1] = self.retrieval_token_idx     
            for i in range(1,self.LenRET+1):
              tokens[-i] = self.retrieval_token_idx[-i]
        else:
          if tokens[-1] not in [self.retrieval_token_idx, self.tokenizer.pad_token_id]:
            tokens[-1] = self.retrieval_token_idx

        return {
            'image_path': image_path,
            'images': images,
            'cap_img': cap_img,
            'tokens': tokens,
            'caption_len': caption_len,
            'tokens_tc': tokens_tc,
            'tc_len': tc_len,
            'new_caption_clip_tokens': new_caption_clip_tokens,
            'tokens_cc': tokens_cc,
            'cc_len': cc_len,
            'pc_len': pc_len,
            'masks_tc': masks_tc,
            'text_condition_clip_tokens':text_condition_clip_tokens
        }
      except Exception as e:
        print(f'Error reading {image_path} with caption {caption}: {e}')
        # Pick a new example at random.
        idx = np.random.randint(0, len(self)-1)
        
