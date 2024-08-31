"""
modified based on FromageModel: https://github.com/kohjingyu/fromage/blob/main/fromage/models.py
"""

from typing import Callable, List, Optional, Tuple, Union
from collections import namedtuple
import json
import glob
import math
import numpy as np
import os
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functools import partial
import pickle as pkl
from PIL import Image, UnidentifiedImageError

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers import OPTForCausalLM, GPT2Tokenizer
from fromage.modified_opt import Modified_OPTForCausalLM

from transformers import CLIPVisionModel, CLIPVisionConfig,CLIPModel
from fromage.modified_clip import Modified_CLIPModel
from fromage import utils
from fromage.transformer_fusion import Transformer_Fusion
import random
from transformers import LlamaForCausalLM, LlamaTokenizer
from fromage.modified_llama import Modified_LlamaForCausalLM


class FrozenArgs:
  freeze_lm: bool = True
  freeze_vm: bool = True
  opt_version: str = 'facebook/opt-6.7b'
  visual_encoder: str = 'openai/clip-vit-large-patch14'
  n_visual_tokens: int = 1
  image_embed_dropout_prob: float = 0.0
  task: str = 'cap'
  shared_emb_dim: Optional[int] = 256
  text_emb_layers: List[int] = [-1]
  retrieval_token_idx: int = 0


class MCLModel(nn.Module):
  def __init__(self, tokenizer, args: FrozenArgs = FrozenArgs()):
    super().__init__()
    self.tokenizer = tokenizer
    self.feature_extractor = utils.get_feature_extractor_for_model(args.visual_encoder, train=False)
    self.image_token = self.tokenizer.cls_token_id
    assert args.text_emb_layers != set(args.text_emb_layers), 'text_emb_layers not unique'
    self.args = args
    try:
      self.LenRET = args.LenRET
    except:
      self.LenRET=1
    opt_version = args.opt_version
    visual_encoder = args.visual_encoder
    n_visual_tokens = args.n_visual_tokens
    
    print(f"Using {visual_encoder} for the visual model with {n_visual_tokens} visual tokens.")

    self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
#     self.logit_scale = torch.ones([]) * np.log(1 / 0.07)
    if self.args.Llama:
      self.lm = Modified_LlamaForCausalLM.from_pretrained("fromage/llama/hugging-llama-2-7b",torch_dtype=torch.float16)
      print(f"Using Llama for the language model.")
    else: 
      if 'facebook/opt' in opt_version:
        self.lm = Modified_OPTForCausalLM.from_pretrained(opt_version,local_files_only=False,torch_dtype=torch.float16)
      print(f"Using {opt_version} for the language model.")
    self.opt_version = opt_version

    if self.args.freeze_lm:
      self.lm.eval()
      print("Freezing the LM.")
      for param in self.lm.parameters():
        param.requires_grad = False

    else:
      self.lm.train()

    self.retrieval_token_idx = args.retrieval_token_idx
    print(f'Initializing embedding for the retrieval token [RET] (id = {self.retrieval_token_idx}).')
    self.lm.resize_token_embeddings(len(tokenizer))

    self.input_embeddings = self.lm.get_input_embeddings()
    for param in self.input_embeddings.parameters():
        param.requires_grad=True

    print("Restoring pretrained weights for the visual model.")
    if 'clip' in visual_encoder:
      self.visual_model = CLIPModel.from_pretrained(visual_encoder,local_files_only=False)
    else:
      self.visual_model = AutoModel.from_pretrained(visual_encoder)

    if 'clip' in visual_encoder:
      hidden_size = self.visual_model.config.projection_dim
    else:
      raise NotImplementedError

    if self.args.freeze_vm:
      print("Freezing the VM.")
      self.visual_model.eval()
      for param in self.visual_model.parameters():
        param.requires_grad = False
        
    else:
      self.visual_model.train()

    self.visual_model_name = visual_encoder

    embedding_dim = self.input_embeddings.embedding_dim * self.args.n_visual_tokens
    self.text_hidden_fcs = nn.ModuleList([])
    if self.args.shared_emb_dim is None:
      if len(self.args.text_emb_layers) == 1:
        if (self.args.text_emb_layers[0] in [-1, self.lm.config.num_hidden_layers]) and ('bert' not in opt_version):
          out_dim = self.lm.config.word_embed_proj_dim
        else:
          out_dim = self.lm.config.hidden_size
      else:
        if (-1 in self.args.text_emb_layers) or (self.lm.config.num_hidden_layers in self.args.text_emb_layers) \
          and (self.lm.config.word_embed_proj_dim != self.lm.config.hidden_size):
          raise ValueError('No projection dim specified but model uses last output layer and an intermediate one (which have different dims).')
        else:
          out_dim = self.lm.config.hidden_size
    else:
      out_dim = self.args.shared_emb_dim

      for layer_idx in self.args.text_emb_layers:
        if (layer_idx == -1 or layer_idx == self.lm.config.num_hidden_layers) and ('bert' not in opt_version):
          if self.args.Llama:
            in_dim = self.lm.config.hidden_size
          else:
            in_dim = self.lm.config.word_embed_proj_dim

          text_fc = [nn.Linear(in_dim, out_dim), nn.Dropout(self.args.text_embed_dropout_prob)]
          self.text_hidden_fcs.append(nn.Sequential(*text_fc))

        elif layer_idx < self.lm.config.num_hidden_layers:
          text_fc = [nn.Linear(self.lm.config.hidden_size, out_dim), nn.Dropout(self.args.text_embed_dropout_prob)]
          self.text_hidden_fcs.append(nn.Sequential(*text_fc))
        else:
          raise ValueError(f'Embedding of layer {layer_idx} was requested but model only has {self.lm.config.num_hidden_layers} layers.')

    self.visual_embeddings = nn.Linear(hidden_size, embedding_dim)
    self.visual_fc = nn.Linear(hidden_size, out_dim)
    self.image_dropout = nn.Dropout(self.args.image_embed_dropout_prob)

    
    self.transformer_fusion = Transformer_Fusion(dim_self=self.args.shared_emb_dim,num_heads=4,num_layers=2)
    self.dropout2d = nn.Dropout2d(p=0.5)
  def get_visual_embs(self, pixel_values: torch.FloatTensor, mode: str = 'cap'):

    # Extract visual embeddings from the vision encoder.
    if 'clip' in self.visual_model_name:
      with torch.no_grad():
        outputs = self.visual_model.get_image_features(pixel_values)
      encoder_outputs = outputs
#       encoder_outputs = outputs.pooler_output
    else:
      raise NotImplementedError

    # Use the correct fc based on function argument.
    if mode == 'cap' or mode == 'mc_cap':
      visual_embs = self.visual_embeddings(encoder_outputs)  # (2, D * n_visual_tokens)
      visual_embs = torch.reshape(visual_embs, (visual_embs.shape[0], self.args.n_visual_tokens, -1))
    elif mode == 'ret':
      visual_embs = encoder_outputs
      visual_embs = torch.reshape(visual_embs, (visual_embs.shape[0], 1, -1))
    else:
      raise NotImplementedError

    visual_embs = self.image_dropout(visual_embs)
    return visual_embs

  def get_text_embs(self, text):
    # Extract visual embeddings from the vision encoder.
    if 'clip' in self.visual_model_name:
      with torch.no_grad():
        outputs = self.visual_model.get_text_features(text)
    text_embs = self.image_dropout(outputs)
    return text_embs

  def train(self, mode=True):
    super(MCLModel, self).train(mode=mode)
    # Overwrite train() to ensure Frozen models remain frozen.
    if self.args.freeze_lm:
      self.lm.eval()
    if self.args.freeze_vm:
      self.visual_model.eval()


  def forward(
    self,
    pixel_values: torch.FloatTensor,
    labels: torch.LongTensor,
    caption_len: torch.LongTensor,
    mode: str = 'cap',
    concat_captions: bool = False,
    input_prefix: Optional[str] = None,
    inference: bool = False,
    tc_tokens = None,
    tc_len=None,
    new_caption_clip_tokens=None,
    pc_len = None,
    masks_tc = None,
    text_condition_clip_tokens = None,
  ):

  
    
    if mode =='mc_ret':
      visual_embs_img = self.get_visual_embs(pixel_values, 'cap')
      
      ## target embedding
      visual_embs = self.get_text_embs(new_caption_clip_tokens)

      batch_size, vis_seq_len, _ = visual_embs_img.shape  # vis_seq_len = n_visual_tokens
      if labels is not None:
        assert labels.shape[0] == batch_size, (visual_embs_img.shape, labels.shape)

      input_embs = self.input_embeddings(labels)  # (N, T, D)
      last_embedding_idx = caption_len - self.LenRET  # -1 to retrieve the token before the eos token
    else:
      visual_embs = self.get_visual_embs(pixel_values, mode)

      batch_size, vis_seq_len, _ = visual_embs.shape  # vis_seq_len = n_visual_tokens
      if labels is not None:
        assert labels.shape[0] == batch_size, (visual_embs.shape, labels.shape)

      input_embs = self.input_embeddings(labels)  # (N, T, D)

      last_embedding_idx = caption_len - self.LenRET  # -1 to retrieve the token before the eos token


    if mode =='mc_ret':
      # Concat to text embeddings.
      condition_seq_len = 0

      # Just add visual embeddings.
      input_embs = torch.cat([visual_embs_img, input_embs], axis=1)
      last_embedding_idx += vis_seq_len
      condition_seq_len += vis_seq_len
      full_labels = torch.zeros(visual_embs_img.shape[:2], dtype=torch.int64).to(visual_embs_img.device) - 100
      add_mask = torch.ones(visual_embs_img.shape[:2], dtype=torch.int64).to(visual_embs_img.device)
    
      # Mask out embedding tokens in the labels.
      full_labels = torch.cat([full_labels, labels], axis=1)
      full_attention_mask = torch.cat([add_mask, masks_tc], axis=1)

      pad_idx = []

      for label in full_labels:
        for k, token in enumerate(label):
          # Mask out retrieval token if it exists.
          if token in [self.tokenizer.pad_token_id]:
            label[k:] = -100
            pad_idx.append(k)
            break
          if k == len(label) - 1:  # No padding found.
            pad_idx.append(k + 1)
      assert len(pad_idx) == batch_size, (len(pad_idx), batch_size)

      attention_mask = full_attention_mask if self.args.masktc else None
        
      output = self.lm(inputs_embeds=input_embs,
                       labels=full_labels,
                       attention_mask = attention_mask,
                       output_hidden_states=True)

    elif mode == 'mc_cap':
      # Concat to text embeddings.
      condition_seq_len = 0

      # Just add visual embeddings.
      input_embs = torch.cat([visual_embs, input_embs], axis=1)
      last_embedding_idx += vis_seq_len
      condition_seq_len += vis_seq_len
      full_labels = torch.zeros(visual_embs.shape[:2], dtype=torch.int64).to(visual_embs.device) - 100
    

      # Mask out embedding tokens in the labels.
      full_labels = torch.cat([full_labels, labels], axis=1)

      pad_idx = []

      for j,label in enumerate(full_labels):
        for k, token in enumerate(label):
          # Mask out retrieval token if it exists.

          if k<pc_len[j]:
            label[k] = -100
          if token in [self.tokenizer.pad_token_id, self.retrieval_token_idx]:
            label[k:] = -100
            pad_idx.append(k)
            break
          if k == len(label) - 1:  # No padding found.
            pad_idx.append(k + 1)
      assert len(pad_idx) == batch_size, (len(pad_idx), batch_size)

      bs, seq_len, embs_dim = input_embs.shape
      output = self.lm(inputs_embeds=input_embs,
                       labels=full_labels,
                       output_hidden_states=True)

    elif mode == 'cap':
      # Concat to text embeddings.
      condition_seq_len = 0


      input_embs = torch.cat([visual_embs, input_embs], axis=1)
      last_embedding_idx += vis_seq_len
      condition_seq_len += vis_seq_len
      full_labels = torch.zeros(visual_embs.shape[:2], dtype=torch.int64).to(visual_embs.device) - 100
      add_mask = torch.ones(visual_embs.shape[:2], dtype=torch.int64).to(visual_embs.device)
   
      full_labels = torch.cat([full_labels, labels], axis=1)
      full_attention_mask = torch.cat([add_mask, masks_tc], axis=1)
      pad_idx = []

      for label in full_labels:
        for k, token in enumerate(label):
          # Mask out retrieval token if it exists.
          if token in [self.tokenizer.pad_token_id, self.retrieval_token_idx]:
            label[k:] = -100
            pad_idx.append(k)
            break
          if k == len(label) - 1:  # No padding found.
            pad_idx.append(k + 1)
      assert len(pad_idx) == batch_size, (len(pad_idx), batch_size)

      bs, seq_len, embs_dim = input_embs.shape
      
      if concat_captions:
        assert len(input_embs.shape) == 3, input_embs
        assert len(full_labels.shape) == 2, full_labels
        assert batch_size % 2 == 0
        all_concat_input_embs = []
        all_concat_labels = []

        # Rearrange embeddings and labels (and their padding) to concatenate captions.
        for i in range(batch_size // 2):
          first_idx = i * 2
          second_idx = first_idx + 1
          first_emb = input_embs[first_idx, :pad_idx[first_idx], :]
          first_labels = full_labels[first_idx, :pad_idx[first_idx]]
          first_padding = input_embs[first_idx, pad_idx[first_idx]:, :]
          first_labels_padding = full_labels[first_idx, pad_idx[first_idx]:]

          second_emb = input_embs[second_idx, :pad_idx[second_idx], :]
          second_labels = full_labels[second_idx, :pad_idx[second_idx]]
          second_padding = input_embs[second_idx, pad_idx[second_idx]:, :]
          second_labels_padding = full_labels[second_idx, pad_idx[second_idx]:]

          assert torch.all(first_labels_padding == -100), first_labels_padding
          assert torch.all(second_labels_padding == -100), second_labels_padding
          concat_input_embs = torch.cat([first_emb, second_emb, first_padding, second_padding], axis=0)   # (T*2, 768)
          concat_labels = torch.cat([first_labels, second_labels, first_labels_padding, second_labels_padding], axis=0)   # (T*2, 768)
          all_concat_input_embs.append(concat_input_embs)
          all_concat_labels.append(concat_labels)

        # Pad to max length.
        input_embs = torch.stack(all_concat_input_embs, axis=0)  # (N/2, T*2, 768)
        full_labels = torch.stack(all_concat_labels, axis=0)  # (N/2, T*2, 768)
        assert input_embs.shape == (bs // 2, seq_len * 2, embs_dim), input_embs.shape
        assert full_labels.shape == (bs // 2, seq_len * 2), full_labels.shape

      output = self.lm(inputs_embeds=input_embs,
                       labels=full_labels,
                       output_hidden_states=True)
    elif mode == 'ret':
      full_labels = torch.clone(labels)
      full_attention_mask = torch.clone(masks_tc)
        
      pad_idx = []
      for label in full_labels:
        for k, token in enumerate(label):
          if token == self.tokenizer.pad_token_id:
            label[k:] = -100
            pad_idx.append(k)
            break
          if k == len(label) - 1:  # No padding found.
            pad_idx.append(k + 1)
      assert len(pad_idx) == batch_size, (len(pad_idx), batch_size)
      attention_mask = full_attention_mask if self.args.masktc else None
      if self.args.masktc:
        output = self.lm(inputs_embeds=input_embs,
                         labels=full_labels,
                         attention_mask = attention_mask,
                         output_hidden_states=True)
    else:
      raise NotImplementedError

    last_embedding = None
    last_output_logit = None
    hidden_states = []

    if mode == 'ret' or mode =='mc_ret':
      if self.args.shared_emb_dim is not None:
        for idx, fc_layer in zip(self.args.text_emb_layers, self.text_hidden_fcs):
          hidden_states.append(fc_layer(output.hidden_states[idx]))  # (N, seq_len, 2048)
      else:
        for idx in self.args.text_emb_layers:
          hidden_states.append(output.hidden_states[idx])

      # Add hidden states together.
      last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

      if not concat_captions:
        if self.LenRET>1:
          last_embedding = torch.stack([last_hidden_state[i, last_embedding_idx[i]:last_embedding_idx[i]+self.LenRET, :] for i in range(batch_size)], axis=0)  # (N,L,D)

          last_embedding = self.transformer_fusion(last_embedding)

          last_embedding = torch.mean(last_embedding, dim=1) # (N,D)
        else:
          last_embedding = torch.stack([last_hidden_state[i, last_embedding_idx[i], :] for i in range(batch_size)], axis=0)  # (N, D)
        
        
        last_output_logit = torch.stack([output.logits[i, last_embedding_idx[i] - 1, :] for i in range(batch_size)], axis=0)  # (N, D)
      else:
        raise NotImplementedError

      # Compute retrieval loss.
      if mode =='ret':
        visual_embs = visual_embs[:, 0, :]
      visual_embs = visual_embs / visual_embs.norm(dim=1, keepdim=True)
      last_embedding = last_embedding / last_embedding.norm(dim=1, keepdim=True)


      logit_scale = 14.2
      visual_embs = logit_scale * visual_embs
    elif mode == 'cap':
      pass
    elif mode == 'mc_cap':
      pass
    else:
      raise NotImplementedError

    return output, full_labels, last_embedding, visual_embs

  def generate(self, embeddings = torch.FloatTensor, max_len: int = 32,
               temperature: float = 0.0, top_p: float = 1.0, min_word_tokens: int = 0,
               ret_scale_factor: float = 1.0, filter_value: float = -float('Inf'), filter_tokens=None):
    """Runs greedy decoding and returns generated captions.

    Args:
      embeddings: Input condition that the model uses for autoregressive generation.
      max_len: Maximum number of tokens to generate.
      temperature: Used to modulate logit distribution.
      top_p: If set to < 1, the smallest set of tokens with highest probabilities that add up to top_p or higher are kept for generation.
      min_word_tokens: Minimum number of words to generate before allowing a [RET] output.
      ret_scale_factor: Proportion to scale [RET] token logits by. A higher value may increase the probability of the model generating [RET] outputs.
      filter_value: Value to assign to tokens that should never be generated.
    Outputs:
      out: (N, T) int32 sequence of output tokens.
      output_embeddings: (N, T, 256) sequence of text output embeddings.
    """
    self.lm.eval()

    with torch.no_grad():  # no tracking history
      batch_size, s, _ = embeddings.shape
      # init output with image tokens
      out = None
      past_key_values = None
      output_embeddings = []
      output_logits = []

      for i in range(max_len):
        if 'opt' in self.opt_version:
          output = self.lm(inputs_embeds=embeddings, use_cache=False, output_hidden_states=True)
        else:
          if i == 0:
            output = self.lm(inputs_embeds=embeddings, use_cache=True, past_key_values=None, output_hidden_states=True)
          else:
            output = self.lm(input_ids=out[:, -1:], use_cache=True, past_key_values=past_key_values, output_hidden_states=True)

        # Collect and sum the hidden states.
        hidden_states = []
        if self.args.shared_emb_dim is not None:
          for idx, fc_layer in zip(self.args.text_emb_layers, self.text_hidden_fcs):
            hidden_states.append(fc_layer(output.hidden_states[idx]))  # (N, seq_len, 2048)
        else:
          for idx in self.args.text_emb_layers:
            hidden_states.append(output.hidden_states[idx])
        # Add hidden states together.
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)  # (N, T, 256)
        last_embedding = last_hidden_state / last_hidden_state.norm(dim=-1, keepdim=True)
        output_embeddings.append(last_embedding)

        logits = output.logits[:, -1, :]  # (N, vocab_size)
        if filter_tokens is not None:
          for filter_token in filter_tokens:
            logits[:, filter_token] = filter_value

        if top_p == 1.0:
          logits = logits.cpu()
        output_logits.append(logits)

        if self.retrieval_token_idx != -1 and self.retrieval_token_idx is not None:
          if i < min_word_tokens:
            # Eliminate probability of generating [RET] if this is earlier than min_word_tokens.
            logits[:, self.retrieval_token_idx] = filter_value
          else:
            # Multiply by scaling factor.
            logits[:, self.retrieval_token_idx] = logits[:, self.retrieval_token_idx] * ret_scale_factor

        past_key_values = output.past_key_values

        if temperature == 0.0:
          if top_p != 1.0:
            raise ValueError('top_p cannot be set if temperature is 0 (greedy decoding).')
          next_token = torch.argmax(logits, keepdim=True, dim=-1)  # (N, 1)
        else:
          logits = logits / temperature

          # Apply top-p filtering.
          if top_p < 1.0:
            assert top_p > 0, f'top_p should be above 0, got {top_p} instead.'
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # (N, D) and (N, D)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1) # (N, D)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            for j in range(sorted_indices.shape[0]):
              indices_to_remove = sorted_indices[j, sorted_indices_to_remove[j, :]]
              logits[j, indices_to_remove] = filter_value

          token_weights = logits.exp()   # (N, vocab_size)
          next_token = torch.multinomial(token_weights, 1)  # (N, 1)

        next_token = next_token.long().to(embeddings.device)
        if out is not None:
          out = torch.cat([out, next_token], dim=-1)
        else:
          out = next_token

        if 'opt' in self.opt_version:
          next_embedding = self.input_embeddings(next_token)
          embeddings = torch.cat([embeddings, next_embedding], dim=1)
        elif (self.tokenizer.eos_token_id and (next_token == self.tokenizer.eos_token_id).all()):
          # End of generation.
          break

    return out, output_embeddings, output_logits


class MCL(nn.Module):
  def __init__(self, tokenizer, model_args: Optional[FrozenArgs] = None):
    super().__init__()
    self.model = MCLModel(tokenizer, model_args)


  def __call__(self, images: Tensor, tgt_tokens: Optional[Tensor] = None, caption_len: Optional[Tensor] = None,
               generate: bool = False, num_words: int = 32, temperature: float = 1.0, top_p: float = 1.0,
               ret_scale_factor: float = 1.0, min_word_tokens: int = 0,
               mode: str = 'cap', concat_captions: bool = False,
               input_prefix: Optional[str] = None, inference: bool = False, tc_tokens = None, tc_len=None, new_caption_clip_tokens=None,pc_len=None,masks_tc=None,text_condition_clip_tokens=None) -> Tensor:
    if generate:
      return self.model.generate(images, num_words, temperature=temperature, top_p=top_p,
                                 min_word_tokens=min_word_tokens, ret_scale_factor=ret_scale_factor)
    else:
      output = self.model(
        pixel_values = images,
        labels = tgt_tokens,
        caption_len = caption_len,
        mode = mode,
        concat_captions = concat_captions,
        input_prefix = input_prefix,
        inference = inference,
        tc_tokens = tc_tokens, 
        tc_len=tc_len, 
        new_caption_clip_tokens=new_caption_clip_tokens,
        pc_len=pc_len,
        masks_tc = masks_tc,
        text_condition_clip_tokens=text_condition_clip_tokens,
      )
      return output


  def generate_for_caption(
    self, prompts: List, num_words: int = 0, ret_scale_factor: float = 1.0, top_p: float = 1.0, temperature: float = 0.0,
    max_num_rets: int = 1, max_img_per_ret: int = 1,filter_tokens=None):
    """
    Encode prompts into embeddings.

    Args:
      prompts: List of interleaved PIL.Image.Image and strings representing input to the model.
      num_words: Maximum number of words to generate for. If num_words = 0, the model will run its forward pass and return the outputs.
      ret_scale_factor: Proportion to scale [RET] token logits by. A higher value may increase the probability of the model generating [RET] outputs.
      top_p: If set to < 1, the smallest set of tokens with highest probabilities that add up to top_p or higher are kept for generation.
      temperature: Used to modulate logit distribution.
      max_num_rets: Maximum number of images to return in one generation pass.
      max_img_per_ret: Maximum number of images to return for each [RET] token.
    Returns:
      return_outputs: List consisting of either str or List[PIL.Image.Image] objects, representing image-text interleaved model outputs.
    """
    input_embs = []
    input_ids = []
    add_bos = True

    for i, p in enumerate(prompts):
      if type(p) == Image.Image:
        # Encode as image.
        pixel_values = utils.get_pixel_values_for_model(self.model.feature_extractor, p)
        pixel_values = pixel_values.to(device=self.model.logit_scale.device, dtype=self.model.logit_scale.dtype)
        pixel_values = pixel_values[None, ...]

        visual_embs = self.model.get_visual_embs(pixel_values, mode='cap')  # (1, n_visual_tokens, D)
        input_embs.append(visual_embs)
      elif type(p) == str:
        text_ids = self.model.tokenizer(p, add_special_tokens=True, return_tensors="pt").input_ids.to(self.model.logit_scale.device)
        if not add_bos:
          # Remove <bos> tag.
          text_ids = text_ids[:, 1:]
        else:
          # Only add <bos> once.
          add_bos = False

        text_embs = self.model.input_embeddings(text_ids)  # (1, T, D)
        input_embs.append(text_embs)
        input_ids.append(text_ids)
      else:
        raise ValueError(f'Input prompts should be either PIL.Image.Image or str types, got {type(p)} instead.')
    input_embs = torch.cat(input_embs, dim=1)
    input_ids = torch.cat(input_ids, dim=1)

    if num_words == 0:
      generated_ids = input_ids
      outputs = self.model.lm(inputs_embeds=input_embs, use_cache=False, output_hidden_states=True)
      # Map outputs to embeddings, so we can retrieve embeddings from the [RET] tokens.
      out = []
      for x, fc in zip(self.model.args.text_emb_layers, self.model.text_hidden_fcs):
          out.append(fc(outputs.hidden_states[x]))
      embeddings = torch.stack(out, dim=-1).sum(dim=-1)
      embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # (N, T, 256)
    elif num_words > 0:
      generated_ids, generated_embeddings, _ = self.model.generate(input_embs, num_words,
        temperature=temperature, top_p=top_p, ret_scale_factor=ret_scale_factor,filter_tokens=filter_tokens)
      embeddings = generated_embeddings[-1][:, input_embs.shape[1]:]

      # Truncate to newline.
      newline_token_id = self.model.tokenizer('\n', add_special_tokens=False).input_ids[0]
      trunc_idx = 0
      for j in range(generated_ids.shape[1]):
        if generated_ids[0, j] == newline_token_id:
          trunc_idx = j
          break
      if trunc_idx > 0:
        generated_ids = generated_ids[:, :trunc_idx]
        embeddings = embeddings[:, :trunc_idx]
    else:
      raise ValueError

    # Save outputs as an interleaved list.
    return_outputs = []

    caption = self.model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return_outputs.append(utils.truncate_caption(caption))
    

    return return_outputs




