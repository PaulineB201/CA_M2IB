import logging
import torch
from torch import nn
from torch.nn import functional as F
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model).to(device)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape: [len, batch, dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class CrossmodalTransformer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        emb_dropout,
        attn_dropout,
        res_dropout,
        relu_dropout,
        n_layer,
        attn_mask,
        scale_embedding=True,
    ):
        super(CrossmodalTransformer, self).__init__()
        self.attn_mask = attn_mask
        self.emb_scale = math.sqrt(d_model) if scale_embedding else 1.0
        self.pos =  PositionalEncoding1D(11).to(device)
        self.pos_im =  PositionalEncoding1D(224).to(device)
        self.emb_dropout = emb_dropout
        self.layers = nn.ModuleList([])
        for layer in range(n_layer):
            new_layer = TransformerEncoderBlock(
                d_model, nhead, d_model * 4, attn_dropout, res_dropout, relu_dropout
            )
            self.layers.append(new_layer)

    def forward(self, x_query, x_key=None, x_key_padding_mask = None):
        # Positional Encoder  -> (batch, len) -> (batch, len, dim)
        x_query_pos = self.pos(x_query.unsqueeze(0))
        
        # (batch, len, dim) -> (len, batch, dim)
        x_query = F.dropout(
            (self.emb_scale * x_query + x_query_pos), self.emb_dropout, self.training
        ).transpose(0, 1)
        
        if x_key is not None:
            x_key_pos = self.pos_im(x_key)
            x_key = F.dropout(
                (self.emb_scale * x_key + x_key_pos), self.emb_dropout, self.training
            ).transpose(0, 1)
        for layer in self.layers:
            x_query = layer(x_query, x_key, attn_mask=self.attn_mask)
        return x_query


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, attn_dropout, res_dropout, relu_dropout):
        """
        Args:
            d_model: number of features in input 
            nhead: number of heads in the multi-head attention models
            dim_feedforward: dimension of feedforward network model 
            attn_dropout: dropout for multihead attention
            res_dropout: dropout for residual model
            relu_dropout: dropout for relu
        """
        super(TransformerEncoderBlock, self).__init__()
        self.transformer = TransformerBlock(d_model, nhead, attn_dropout, res_dropout)
        self.feedforward = FeedForwardBlock(d_model, dim_feedforward, res_dropout, relu_dropout)
    def forward(self, x_query, x_key=None, x_key_padding_mask=None, attn_mask=None):
   
        if x_key is not None:
            x = self.transformer(
                x_query, x_key, x_key, key_padding_mask=x_key_padding_mask, attn_mask=attn_mask
            )
        else:
            x = self.transformer(
                x_query, x_query, x_query, key_padding_mask=x_key_padding_mask, attn_mask=attn_mask,
            )
        x = self.feedforward(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, attn_dropout, res_dropout):
        super(TransformerBlock, self).__init__()
        self.res_dropout = res_dropout
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=attn_dropout)
        self.layernorm = nn.LayerNorm(d_model).to(device)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=True):
        mask = self.get_future_mask(query, key) if attn_mask else None
        query, key, value = [self.layernorm(x) for x in (query, key, value)]
        x = self.self_attn(query, key, value, key_padding_mask=key_padding_mask, attn_mask=mask)[0]
        x = query + F.dropout(x, self.res_dropout, self.training)
        return x
    
    def get_future_mask(self, q, k=None):
        dim_query = q.shape[0]
        dim_key = dim_query if k is None else k.shape[0]
        future_mask = torch.triu(torch.ones(dim_query, dim_key, device = q.device), diagonal=1).float()
        future_mask = future_mask.masked_fill(future_mask == float(1), float('-inf'))
        return future_mask

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, dim_feedforward, res_dropout, relu_dropout):
        super(FeedForwardBlock, self).__init__()
        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.layernorm(x)
        x2 = self.linear2(F.dropout(F.relu(self.linear1(x)), self.relu_dropout, self.training))
        x = x + F.dropout(x2, self.res_dropout, self.training)
        return x


class ca_layer(nn.Module):
    def __init__(
        self,
        orig_d_a = 224,
        orig_d_t = 10,
        n_head = 8,
        n_cmlayer = 4,
        d_out = 7,
        only_image_dim = 224,
        d_model=11,
        emb_dropout=0.25,
        attn_dropout=0.1,
        attn_dropout_image=0.0,
        attn_dropout_vision=0.0,
        relu_dropout=0.1,
        res_dropout=0.1,
        out_dropout=0.0,
        max_position=128,
        attn_mask=True,
        scale_embedding=True,
    ):
        super(ca_layer, self).__init__()

        self.d_model = d_model
        self.emb_dropout = emb_dropout
        self.out_dropout = out_dropout


        self.image_layers_with_text = CrossmodalTransformer(
                d_model,
                n_head,
                emb_dropout,
                attn_dropout_image,
                res_dropout,
                relu_dropout,
                n_cmlayer,
                attn_mask,
        )

        self.text_layers_with_image = CrossmodalTransformer(
                d_model,
                n_head,
                emb_dropout,
                attn_dropout,
                res_dropout,
                relu_dropout,
                n_cmlayer,
                attn_mask,
        )

        self.image_layers = CrossmodalTransformer(
            d_model,
            n_head,
            emb_dropout,
            attn_dropout,
            res_dropout,
            relu_dropout,
            n_cmlayer,
            attn_mask,
        )

        self.text_layers = CrossmodalTransformer(
            d_model,
            n_head,
            emb_dropout,
            attn_dropout,
            res_dropout,
            relu_dropout,
            n_cmlayer,
            attn_mask,
        )


    def forward(self, x_image, x_text =None,
                      a_mask =None,
                      t_mask =None,
                       ):
        """
            Args:
        x_image, x_text : input tensor -> (batch, len, dim)
        """

        x_image = x_image[:,0,:,:].squeeze(1) 

        # Crossmodal Attention
        # output: (len, batch, dim)
        x_text_with_image = self.text_layers_with_image(x_text,
                                                        x_image,
                                                        x_key_padding_mask=a_mask)
        
        x_image_with_text = self.image_layers_with_text(x_image,x_text)

        x_image2 = x_image_with_text
        x_text2 = x_text_with_image

        x_image2 = self.image_layers(x_image2)
        x_text2 = self.text_layers(x_text2)
            
        return x_text2,x_image2