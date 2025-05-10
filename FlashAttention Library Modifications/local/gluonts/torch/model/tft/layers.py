# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from gluonts.core.component import validated
from gluonts.torch.modules.feature import (
    FeatureEmbedder as BaseFeatureEmbedder,
)



import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.nn.attention import sdpa_kernel, SDPBackend


class PyTorchFlashMHA(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        causal: bool = False,
        max_seq_length: int = 2048,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal

        self.to_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Optional causal mask buffer if you want to reuse it
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(max_seq_length, max_seq_length, dtype=torch.bool),
                diagonal=1,
            ).view(1, 1, max_seq_length, max_seq_length),
        )

    def forward(
        self,
        x: torch.Tensor,                        # [B, T, C]
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, T] (1=keep, 0=pad)
    ) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # reshape to [B, H, T, D]
        q, k, v = [
            t.view(B, T, self.num_heads, self.head_dim)
             .transpose(1, 2)
            for t in qkv
        ]
        q = q * self.scale

        # build attn_mask combining causal + pad
        attn_mask = None
        if self.causal:
            attn_mask = self.causal_mask[:, :, :T, :T]
        if key_padding_mask is not None:
            pad_mask = (~key_padding_mask.bool()).view(B, 1, 1, T)
            attn_mask = pad_mask if attn_mask is None else (attn_mask | pad_mask)

        # fused flash or fallback
        # with torch.backends.cuda.sdp_kernel(
        #     enable_flash=True,
        #     enable_math=False,
        #     enable_mem_efficient=True,
        # ):
        #     out = F.scaled_dot_product_attention(
        #         q, k, v,
        #         attn_mask=attn_mask,
        #         dropout_p=self.training and self.to_out[1].p or 0.0,
        #         is_causal=False,   # we’ve already masked if needed
        #     )
            
        # fused flash or fallback via new API
        #with sdpa_kernel(
        #     [SDPBackend.FLASH_ATTENTION],
        #     set_priority=False  # you can also pass other backends here
        #):
         #   out = F.scaled_dot_product_attention(
          #      q, k, v,
           #     attn_mask=attn_mask,
            #    dropout_p=(self.training and self.to_out[1].p) or 0.0,
             #   is_causal=False,   # manual mask already applied
            #)
        
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,  # if you’re using padding
            is_causal=self.causal,              # if you need causal masking
            dropout_p=(self.training and self.to_out[1].p) or 0.0,
        )
        
        
        # out: [B, H, T, D] → [B, T, C]
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.to_out(out)






class FeatureEmbedder(BaseFeatureEmbedder):
    def forward(self, features: torch.Tensor) -> List[torch.Tensor]:  # type: ignore
        concat_features = super().forward(features=features)

        if self._num_features > 1:
            return torch.chunk(concat_features, self._num_features, dim=-1)
        else:
            return [concat_features]


class FeatureProjector(nn.Module):
    @validated()
    def __init__(
        self,
        feature_dims: List[int],
        embedding_dims: List[int],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        assert len(feature_dims) > 0, "Expected len(feature_dims) > 1"
        assert len(feature_dims) == len(
            embedding_dims
        ), "Length of `feature_dims` and `embedding_dims` should match"
        assert all(
            c > 0 for c in feature_dims
        ), "Elements of `feature_dims` should be > 0"
        assert all(
            d > 0 for d in embedding_dims
        ), "Elements of `embedding_dims` should be > 0"

        self.feature_dims = feature_dims
        self._num_features = len(feature_dims)

        self._projectors = nn.ModuleList(
            [
                nn.Linear(out_features=d, in_features=c)
                for c, d in zip(feature_dims, embedding_dims)
            ]
        )

    def forward(self, features: torch.Tensor) -> List[torch.Tensor]:
        """

        Parameters
        ----------
        features
            Numerical features with shape (..., sum(self.feature_dims)).

        Returns
        -------
        projected_features
            List of project features, with shapes
            [(..., self.embedding_dims[i]) for i in self.embedding_dims]
        """
        if self._num_features > 1:
            feature_slices = torch.split(features, self.feature_dims, dim=-1)
        else:
            feature_slices = tuple([features])

        return [
            proj(feat_slice)
            for proj, feat_slice in zip(self._projectors, feature_slices)
        ]


class GatedLinearUnit(nn.Module):
    @validated()
    def __init__(self, dim: int = -1, nonlinear: bool = True):
        super().__init__()
        self.dim = dim
        self.nonlinear = nonlinear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = torch.chunk(x, chunks=2, dim=self.dim)
        if self.nonlinear:
            value = torch.tanh(value)
        gate = torch.sigmoid(gate)
        return gate * value


class GatedResidualNetwork(nn.Module):
    @validated()
    def __init__(
        self,
        d_hidden: int,
        d_input: Optional[int] = None,
        d_output: Optional[int] = None,
        d_static: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_input = d_input or d_hidden
        self.d_static = d_static or 0
        if d_output is None:
            self.d_output = self.d_input
            self.add_skip = False
        else:
            self.d_output = d_output
            if d_output != self.d_input:
                self.add_skip = True
                self.skip_proj = nn.Linear(
                    in_features=self.d_input,
                    out_features=self.d_output,
                )
            else:
                self.add_skip = False

        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=self.d_input + self.d_static,
                out_features=self.d_hidden,
            ),
            nn.ELU(),
            nn.Linear(
                in_features=self.d_hidden,
                out_features=self.d_hidden,
            ),
            nn.Dropout(p=dropout),
            nn.Linear(
                in_features=self.d_hidden,
                out_features=self.d_output * 2,
            ),
            GatedLinearUnit(nonlinear=False),
        )
        self.layer_norm = nn.LayerNorm([self.d_output])

    def forward(
        self, x: torch.Tensor, c: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.add_skip:
            skip = self.skip_proj(x)
        else:
            skip = x
        if self.d_static > 0 and c is None:
            raise ValueError("static variable is expected.")
        if self.d_static == 0 and c is not None:
            raise ValueError("static variable is not accepted.")
        if c is not None:
            x = torch.concat([x, c], dim=-1)
        x = self.mlp(x)
        x = self.layer_norm(x + skip)
        return x


class VariableSelectionNetwork(nn.Module):
    @validated()
    def __init__(
        self,
        d_hidden: int,
        num_vars: int,
        dropout: float = 0.0,
        add_static: bool = False,
    ) -> None:
        super().__init__()
        self.d_hidden = d_hidden
        self.num_vars = num_vars
        self.add_static = add_static

        self.weight_network = GatedResidualNetwork(
            d_hidden=self.d_hidden,
            d_input=self.d_hidden * self.num_vars,
            d_output=self.num_vars,
            d_static=self.d_hidden if add_static else None,
            dropout=dropout,
        )
        self.variable_networks = nn.ModuleList(
            [
                GatedResidualNetwork(d_hidden=d_hidden, dropout=dropout)
                for _ in range(num_vars)
            ]
        )

    def forward(
        self,
        variables: List[torch.Tensor],
        static: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        flatten = torch.cat(variables, dim=-1)
        if static is not None:
            static = static.expand_as(variables[0])
        weight = self.weight_network(flatten, static)
        weight = torch.softmax(weight.unsqueeze(-2), dim=-1)

        var_encodings = [
            net(var) for var, net in zip(variables, self.variable_networks)
        ]
        var_encodings = torch.stack(var_encodings, dim=-1)

        var_encodings = torch.sum(var_encodings * weight, dim=-1)

        return var_encodings, weight


class TemporalFusionEncoder(nn.Module):
    @validated()
    def __init__(
        self,
        d_input: int,
        d_hidden: int,
    ):
        super().__init__()

        self.encoder_lstm = nn.LSTM(
            input_size=d_input, hidden_size=d_hidden, batch_first=True
        )
        self.decoder_lstm = nn.LSTM(
            input_size=d_input, hidden_size=d_hidden, batch_first=True
        )

        self.gate = nn.Sequential(
            nn.Linear(in_features=d_hidden, out_features=d_hidden * 2),
            nn.GLU(),
        )
        if d_input != d_hidden:
            self.skip_proj = nn.Linear(
                in_features=d_input, out_features=d_hidden
            )
            self.add_skip = True
        else:
            self.add_skip = False

        self.lnorm = nn.LayerNorm(d_hidden)

    def forward(
        self,
        ctx_input: torch.Tensor,
        tgt_input: Optional[torch.Tensor] = None,
        states: Optional[List[torch.Tensor]] = None,
    ):
        ctx_encodings, states = self.encoder_lstm(ctx_input, states)

        if tgt_input is not None:
            tgt_encodings, _ = self.decoder_lstm(tgt_input, states)
            encodings = torch.cat((ctx_encodings, tgt_encodings), dim=1)
            skip = torch.cat((ctx_input, tgt_input), dim=1)
        else:
            encodings = ctx_encodings
            skip = ctx_input

        if self.add_skip:
            skip = self.skip_proj(skip)
        encodings = self.gate(encodings)
        encodings = self.lnorm(skip + encodings)
        return encodings


class TemporalFusionDecoder(nn.Module):
    @validated()
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        d_hidden: int,
        d_var: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length

        self.enrich = GatedResidualNetwork(
            d_hidden=d_hidden,
            d_static=d_var,
            dropout=dropout,
        )

        # self.attention = nn.MultiheadAttention(
        #     embed_dim=d_hidden,
        #     num_heads=num_heads,
        #     dropout=dropout,
        #     batch_first=True,
        # )

        # self.attention = MHA(embed_dim=d_hidden,
        #                      num_heads=num_heads,
        #                      dropout=dropout,causal=False  # TFT is not causal, change to True if needed
        #                      )
        # self.attention = MHA(embed_dim=d_hidden,
        #                     num_heads=num_heads,
        #                     use_flash_attn = True,
        #                    dropout=dropout,
        #                    causal=False,  # TFT is not causal, change to True if needed "test with True" ,"cross_attn = True", dtype=torch.float16 / dtype=torch.bfloat16(a100)
        #                     device="cuda"
        #                     )

        self.attention = PyTorchFlashMHA(
            embed_dim=d_hidden,
            num_heads=num_heads,
            dropout=dropout,
            causal=False,
            max_seq_length=context_length + prediction_length,
           )

        self.att_net = nn.Sequential(
            nn.Linear(in_features=d_hidden, out_features=d_hidden * 2),
            nn.GLU(),
        )
        self.att_lnorm = nn.LayerNorm(d_hidden)

        self.ff_net = nn.Sequential(
            GatedResidualNetwork(d_hidden=d_hidden, dropout=dropout),
            nn.Linear(in_features=d_hidden, out_features=d_hidden * 2),
            nn.GLU(),
        )
        self.ff_lnorm = nn.LayerNorm(d_hidden)

    def forward(
        self,
        x: torch.Tensor,
        static: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        expanded_static = static.repeat(
            (1, self.context_length + self.prediction_length, 1)
        )

        skip = x[:, self.context_length :, ...]
        x = self.enrich(x, expanded_static)

        mask_pad = torch.ones_like(mask)[:, 0:1, ...]
        mask_pad = mask_pad.repeat((1, self.prediction_length))
        key_padding_mask = (1.0 - torch.cat((mask, mask_pad), dim=1)).bool()

        query_key_value = x
        # attn_output, _ = self.attention(
        #     query=query_key_value[:, self.context_length :, ...],
        #     key=query_key_value,
        #     value=query_key_value,
        #     key_padding_mask=key_padding_mask,
        # )

        # FlashAttention MHA expects [B, T, C] without key_padding_mask logic
        # attn_input = query_key_value
        # attn_output = self.attention(attn_input)[:, self.context_length :, :]
        
        # cast to half‐precision for FlashAttention, then back to original dtype
        #orig_dtype = attn_input.dtype
        #attn_input_fp16 = attn_input.to(torch.float16)
        #attn_output = (self.attention(attn_input_fp16)[:, self.context_length :, :].to(orig_dtype))
        
        # pass the full sequence + padding mask
        attn_full = self.attention(
            query_key_value,
            key_padding_mask=key_padding_mask,
        )  # [B, T, d_hidden]
        attn_output = attn_full[:, self.context_length :, :]
        
        
        
        att = self.att_net(attn_output)

        x = x[:, self.context_length :, ...]
        x = self.att_lnorm(x + att)
        x = self.ff_net(x)
        x = self.ff_lnorm(x + skip)

        return x
