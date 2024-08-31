import numpy as np
import collections
import copy
import json
import os
import torch
from transformers import logging
from tqdm import tqdm

from PIL import Image
import matplotlib.pyplot as plt



from transformers import AutoFeatureExtractor

import torch
from PIL import Image
import requests
from torch import nn
from transformers import CLIPProcessor, CLIPModel,CLIPTextModel,CLIPVisionModel

from transformers.models.clip.modeling_clip import CLIPTextTransformer

def _make_causal_mask(
    input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


  
from typing import Any, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling

# class Modified_CLIPTextTransformer()
class Modified_CLIPTextTransformer(CLIPTextTransformer):
    def __init__(self, config):
        super().__init__(config)
#         self.linear = nn.Linear(input_dim, output_dim)

    def forward_hidden(
        self,
        llm_hidden_states = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if input_ids is None:
#             raise ValueError("You have to specify input_ids")

#         input_shape = input_ids.size()
#         input_ids = input_ids.view(-1, input_shape[-1])
        prefix_idx = torch.tensor([[49406,   320,  1125,   539]])
        prefix_idx = prefix_idx.to(llm_hidden_states.device)
        prefix_emb = self.embeddings(input_ids=prefix_idx, position_ids=position_ids)
        prefix_emb = prefix_emb.repeat(llm_hidden_states.shape[0],1,1)
        hidden_states = llm_hidden_states
#         hidden_states = torch.cat([prefix_emb,hidden_states],dim=1)
        input_shape = hidden_states.shape[:2]
        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _make_causal_mask(input_shape, hidden_states.dtype, device=hidden_states.device)
        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)
#         pooled_output = [last_hidden_state.shape[0]]
#         print(last_hidden_state.shape)
        pooled_output = last_hidden_state[:,-1,:]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
      
      
# class Modified_CLIPTextTransformer()
class Modified_CLIPModel(CLIPModel):
    def __init__(self, config):
        super().__init__(config)
        text_config = config.text_config
        self.text_model = Modified_CLIPTextTransformer(text_config)
#         self.linear = nn.Linear(input_dim, output_dim)
    def get_hidden_features(
            self,
            llm_hidden_states = None,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> torch.FloatTensor:
            r"""
            Returns:
                text_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The text embeddings obtained by
                applying the projection layer to the pooled output of [`CLIPTextModel`].

            Examples:

            ```python
            >>> from transformers import AutoTokenizer, CLIPModel

            >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

            >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
            >>> text_features = model.get_text_features(**inputs)
            ```"""
            # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            text_outputs = self.text_model.forward_hidden(
                llm_hidden_states=llm_hidden_states,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            pooled_output = text_outputs[1]
            text_features = self.text_projection(pooled_output)

            return text_features