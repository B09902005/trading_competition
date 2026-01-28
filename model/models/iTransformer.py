'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        
        self.use_covariate = configs.use_covariate
        self.use_general_daily = configs.use_general_daily
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        if self.task_name == 'forecast':
            self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        elif self.task_name == 'regression':
            self.flatten_head = nn.Linear(configs.d_model, 1, bias=True)
            self.projector = nn.Linear(configs.n_features, 1, bias=True )

    def forecast(self, x_enc, x_mark_enc, batch_general_daily):

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        x_mark_enc = x_mark_enc if self.use_covariate else None
        batch_general_daily = batch_general_daily if self.use_general_daily else None
        
        inputs = [x for x in [x_enc, x_mark_enc, batch_general_daily] if x is not None]

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(*inputs) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        return dec_out

    def regression(self, x_enc, x_mark_enc, batch_general_daily):
        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        x_mark_enc = x_mark_enc if self.use_covariate else None
        batch_general_daily = batch_general_daily if self.use_general_daily else None
        
        inputs = [x for x in [x_enc, x_mark_enc, batch_general_daily] if x is not None]

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(*inputs) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        # Choose the N - 1 th (the target)
        
        # B N E -> B N 1
        dec_out = self.flatten_head(enc_out).squeeze(-1)
        dec_out = self.projector(dec_out).squeeze(-1)
        
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_general_daily, mask=None):
        if self.task_name == 'forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, batch_general_daily)
            dec_out = dec_out[:, -self.pred_len:, :]
        elif self.task_name == "regression":
            dec_out = self.regression(x_enc, x_mark_enc, batch_general_daily)
        return dec_out
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np

class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        
        self.use_covariate = configs.use_covariate
        self.use_general_daily = configs.use_general_daily
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        self.projector_forecast = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        self.flatten_head = nn.Linear(configs.d_model, 1, bias=True)
        self.projector = nn.Linear(configs.n_features, 1, bias=True )
        if ("classification" in self.task_name):
            self.projector = nn.Linear(configs.n_features, 3, bias=True)

    def forecast(self, x_enc, x_mark_enc, batch_general_daily):

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        x_mark_enc = x_mark_enc if self.use_covariate else None
        batch_general_daily = batch_general_daily if self.use_general_daily else None
        
        inputs = [x for x in [x_enc, x_mark_enc, batch_general_daily] if x is not None]

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(*inputs) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        return dec_out

    def regression(self, x):
        _, _, N = x.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        # Choose the N - 1 th (the target)
        
        # B N E -> B N 1
        dec_out = self.flatten_head(enc_out).squeeze(-1)
        dec_out = self.projector(dec_out).squeeze(-1)
        
        return dec_out

    def classification(self, x, num_choices):
        _, _, N = x.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        # Choose the N - 1 th (the target)
        
        # B N E -> B N 1
        dec_out = self.flatten_head(enc_out).squeeze(-1)
        dec_out = self.projector(dec_out).squeeze(-1)
        return dec_out


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, batch_general_daily, mask=None):
        if ("classification" in self.task_name):
            dec_out = self.classification(x_enc, 3)
        else:
            dec_out = self.regression(x_enc)
        return dec_out 