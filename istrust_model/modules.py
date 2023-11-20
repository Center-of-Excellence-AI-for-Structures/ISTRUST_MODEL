import torch
import torch.nn as nn
import math
import numpy as np
from typing import Tuple
from typing import Optional
from torch import Tensor
import warnings


class Patching(nn.Module):
    """
        Converts a sequence of images into multiple patches
    """
    def __init__(self, image_width: int, image_height: int, patch_width: int, patch_height: int):
        super().__init__()
        if image_width % patch_width != 0:
            raise ValueError('image width is not divisible by patch width')
        if image_height % patch_height != 0:
            raise ValueError('image height is not divisible by patch height')

        self.patch_width = patch_width
        self.patch_height = patch_height

    def forward(self, x):
        # patch images
        x = x.unfold(-2, self.patch_width, self.patch_width).unfold(-2, self.patch_height, self.patch_height)
        # move feature dimension
        x = torch.movedim(x, -5, -3)
        return x


def conv_block(in_planes, out_planes, dropout, kernel_size=3, stride=1, padding=1, bias=False):
    """
        Convolutional block used for ConvPatchEmbed
        Edited from https://github.com/facebookresearch/xcit/blob/main/detection/backbone/xcit.py

        Consists of:
        - single convolutional layer
        - batchnorm
        - dropout
    """
    if dropout is not None:
        return torch.nn.Sequential(
            nn.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
            ),
            nn.BatchNorm2d(out_planes),
            nn.Dropout(p=dropout),
        )
    else:
        return torch.nn.Sequential(
            nn.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
            ),
            nn.BatchNorm2d(out_planes)
        )


class ConvPatchEmbed(nn.Module):
    """
        Convolutional patch embedding using multiple convolutional blocks from conv_block function
        Edited from https://github.com/facebookresearch/xcit/blob/main/detection/backbone/xcit.py
    """
    def __init__(self, image_width=224, image_height=224, patch_width=16, patch_height=16,
                 in_chans=3, d_model=768, dropout=None):
        super().__init__()
        img_size = tuple((image_height, image_width))
        patch_size = tuple((patch_height, patch_width))
        num_patches = (img_size[1] // patch_size[0]) * (img_size[0] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        if patch_size[0] != patch_size[1]:
            raise ValueError("patch width and patch height must be equal")

        if patch_size[0] == 32:
            self.proj = torch.nn.Sequential(
                conv_block(in_chans, d_model // 8, dropout, 7, 4, 3),
                nn.GELU(),
                conv_block(d_model // 8, d_model // 4, dropout, 3, 2, 1),
                nn.GELU(),
                conv_block(d_model // 4, d_model // 2, dropout, 3, 2, 1),
                nn.GELU(),
                conv_block(d_model // 2, d_model, dropout, 3, 2, 1),
            )
        elif patch_size[0] == 16:
            self.proj = torch.nn.Sequential(
                conv_block(in_chans, d_model // 8, dropout, 7, 2, 3),
                nn.GELU(),
                conv_block(d_model // 8, d_model // 4, dropout, 3, 2, 1),
                nn.GELU(),
                conv_block(d_model // 4, d_model // 2, dropout, 3, 2, 1),
                nn.GELU(),
                conv_block(d_model // 2, d_model, dropout, 3, 2, 1),
            )
        else:
            raise NotImplementedError("For convolutional projection, "
                                      "the only supported patch sizes are 16x16 and 32x32")

    def forward(self, x):
        x_shape = x.shape
        # Merge batch dimension, temporal dimension and spatial dimensions into a single batch dimension
        x = torch.flatten(x, 0, 3)
        # Do the actual convolutional embedding
        x = self.proj(x)
        # Put the original batch dimensions back where they were
        x = x.flatten(2).transpose(1, 2)
        x = x.reshape(*x_shape[:4], -1)
        return x


class PositionalEncoding1D(nn.Module):
    """
        1D Positional encoding

        Edited from https://pytorch.org/tutorials/beginner/transformer_tutorial.html

        For spatial positional encoding n_sequence is n_patches
        For temporal positional encoding n_sequence is number of frames per patch/image
    """
    def __init__(self, d_model: int, n_sequence: int = 100):
        super().__init__()
        position = torch.arange(n_sequence).unsqueeze(1)
        # noinspection PyTypeChecker
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(n_sequence, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Args:
                x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return self.pe[:x.size(-2)]


def get_emb(sin_inp):
    """
        Gets a base embedding for one dimension with sin and cos intertwined

        Taken from
        https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/torch_encodings.py
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding2D(nn.Module):
    """
        2D (Spatial) positional encoding

        Taken from
        https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/torch_encodings.py
    """
    def __init__(self, d_model):
        """
            param d_model: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = d_model
        d_model = int(np.ceil(d_model / 4) * 2)
        self.channels = d_model
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
            param tensor: A 4d tensor of size (batch_size, x, y, ch)
            return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels: 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc


class ScaledDotProductAttention(nn.Module):
    """
        Scaled dot-product attention

        Edited from
        https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.temporal_fusion_transformer.
        sub_modules.ScaledDotProductAttention.html#pytorch_forecasting.models.temporal_fusion_transformer.
        sub_modules.ScaledDotProductAttention

        Parameters:
            d_k: int
                hidden model size of queries and keys
            d_k_learnable: bool
                if True, d_k is a learnable parameter
            dropout: float
                dropout applied to the attention weights
            scale: bool
                determines if the attention weights should be scaled by dividing by d_k (if False d_k is not used)
    """
    def __init__(self, d_k: float, d_k_learnable: bool = False, dropout: float = None, scale: bool = True):
        super(ScaledDotProductAttention, self).__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = dropout

        if d_k_learnable is False:
            self.register_buffer('temp_softmax', torch.Tensor([d_k]))
        else:
            self.temp_softmax = nn.Parameter(torch.Tensor([d_k]))

        self.softmax = nn.Softmax(dim=2)
        self.scale = scale

    def forward(self, q, k, v, mask=None):
        """
            Compute scaled dot-product attention

            inputs:
                q,k,v have shape [batch size, number of q/k/v, d_model]
                mask has shape [number of q, number of k] where non-masked
                    attention weights are represented in the mask with a '0'
                    and masked attention weights with '-torch.inf'

            returns:
                output: the actual output of the attention
                attn: the attention weights
        """
        attn = torch.bmm(q, k.permute(0, 2, 1))  # query-key overlap

        if self.scale:
            attn = attn / torch.sqrt(self.temp_softmax)

        if mask is not None:
            attn = attn + mask

        if torch.nan in attn:
            warnings.warn('nan value found in attention weights')

        attn = self.softmax(attn)

        if self.dropout is not None:
            attn = self.dropout(attn)

        output = torch.bmm(attn, v)
        return output, attn


class InterpretableMultiHeadTokenAttention(nn.Module):
    """
        Interpretable multi-head token-attention

        Edited from:
        https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.temporal_fusion_transformer.
        sub_modules.InterpretableMultiHeadAttention.html#pytorch_forecasting.models.temporal_fusion_transformer.sub_modules.InterpretableMultiHeadAttention

        Parameters:
            n_head: int
                number of heads
            d_model: int
                model size
            dropout: float
                dropout probability, used on the attention weights of ScaledDotProductAttention
                and the output of the multi-head attention
            d_value: int
                (hidden) size of the values
    """
    def __init__(self, n_head: int, d_model: int = 128, dropout: float = None, d_value: int = 128):
        super(InterpretableMultiHeadTokenAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = dropout

        self.v_layer = nn.Linear(self.d_model, d_value)
        self.q_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_k) for _ in range(self.n_head)])
        self.k_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_k) for _ in range(self.n_head)])

        self.attention = ScaledDotProductAttention(dropout=dropout, d_k=self.d_k, d_k_learnable=False, scale=True)
        # Learnable matrix used to bring value size (d_value) back to the model size (d_model)
        self.w_h = nn.Linear(d_value, self.d_model, bias=False)

    def forward(self, q_token, k, v, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # In heads, the outputs of the individual heads will be stored
        heads = []
        # In attns, the attention weights of the individual heads will be stored
        attns = []
        # Convert the values to the hidden size d_value
        vs = self.v_layer(v)
        # Repeat the learnable token query over the batch dimension
        q_token = torch.repeat_interleave(q_token, k.shape[0], dim=0)

        # Compute multi-head attention
        for i in range(self.n_head):
            qs = self.q_layers[i](q_token)
            ks = self.k_layers[i](k)
            head, attn = self.attention(qs, ks, vs, mask)
            heads.append(head)
            attns.append(attn)

        # Convert list of heads and attns to a torch Tensor
        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        attn = torch.stack(attns, dim=2)

        # Average over all heads for interpretability
        outputs = torch.mean(head, dim=2) if self.n_head > 1 else head
        # Reshape hidden size (d_value) back to the model size (d_model)
        outputs = self.w_h(outputs)
        if self.dropout is not None:
            outputs = self.dropout(outputs)
        return outputs, attn


class InterpretableMultiHeadSelfAttention(nn.Module):
    """
        Interpretable multi-head self-attention

        Edited from:
        https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.temporal_fusion_transformer.
        sub_modules.InterpretableMultiHeadAttention.html#pytorch_forecasting.models.temporal_fusion_transformer.sub_modules.InterpretableMultiHeadAttention

        Parameters:
            n_head: int
                number of heads
            d_model: int
                model size
            dropout: float
                dropout probability, used on the attention weights of ScaledDotProductAttention
                and the output of the multi-head attention
            d_value: int
                (hidden) size of the values
    """
    def __init__(self, n_head: int, d_model: int, dropout: float = 0.0, d_value: int = 128):
        super(InterpretableMultiHeadSelfAttention, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = self.d_q = self.d_v = d_model // n_head

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = dropout

        self.v_layer = nn.Linear(self.d_model, d_value)
        self.q_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_k) for _ in range(self.n_head)])
        self.k_layers = nn.ModuleList([nn.Linear(self.d_model, self.d_k) for _ in range(self.n_head)])

        self.attention = ScaledDotProductAttention(dropout=dropout, d_k=self.d_k, d_k_learnable=True, scale=True)
        # Learnable matrix used to bring value size (d_value) back to the model size (d_model)
        self.w_h = nn.Linear(d_value, self.d_model, bias=False)

    def forward(self, q, k, v, mask=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # In heads, the outputs of the individual heads will be stored
        heads = []
        # In attns, the attention weights of the individual heads will be stored
        attns = []
        # Convert the values to the hidden size d_value
        vs = self.v_layer(v)

        # Compute multi-head attention
        for i in range(self.n_head):
            qs = self.q_layers[i](q)
            ks = self.k_layers[i](k)
            head, attn = self.attention(qs, ks, vs, mask)
            heads.append(head)
            attns.append(attn)

        # Convert list of heads and attns to a torch Tensor
        head = torch.stack(heads, dim=2) if self.n_head > 1 else heads[0]
        attn = torch.stack(attns, dim=2) if self.n_head > 1 else attns[0]

        # Average over all heads for interpretability
        outputs = torch.mean(head, dim=2) if self.n_head > 1 else head
        # Reshape hidden size (d_value) back to the model size (d_model)
        outputs = self.w_h(outputs)
        if self.dropout is not None:
            outputs = self.dropout(outputs)
        return outputs, attn


class InterpretableTransformerEncoder(nn.Module):
    """
        Interpretable transformer-encoder

        Parameters:
            d_model: int
                model size
            n_head: int
                number of heads
            dropout: float
                dropout probability used in:
                - attention weights of ScaledDotProductAttention
                - output of InterpretableMultiHeadSelfAttention and InterpretableMultiHeadTokenAttention
                - hidden layer of the MLP
    """
    def __init__(self, d_model, n_head, dropout: float = 0.0, layer_norm_eps: float = 1e-5, d_value: int = 128):
        super().__init__()
        self.learnable_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.query = nn.Sequential(nn.Linear(d_model, d_model))
        self.key = nn.Sequential(nn.Linear(d_model, d_model))
        self.value = nn.Sequential(nn.Linear(d_model, d_model))
        self.query_token = nn.Sequential(nn.Linear(d_model, d_model))
        self.key_token = nn.Sequential(nn.Linear(d_model, d_model))
        self.value_token = nn.Sequential(nn.Linear(d_model, d_model))
        self.multi_head_self_attention = InterpretableMultiHeadSelfAttention(
            n_head, d_model, dropout=dropout, d_value=d_value
        )
        self.norm_self = torch.nn.LayerNorm(d_model, layer_norm_eps)
        self.multi_head_token_attention = InterpretableMultiHeadTokenAttention(
            n_head, d_model, dropout=dropout, d_value=d_value
        )
        self.norm_token = torch.nn.LayerNorm(d_model, layer_norm_eps)
        if dropout is not None:
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(p=dropout), nn.Linear(d_model * 4, d_model)
            )
        else:
            self.mlp = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        self.norm_mlp = torch.nn.LayerNorm(d_model, layer_norm_eps)

    def forward(self, x, mask: Optional[Tensor] = None, mask_token: Optional[Tensor] = None):
        # pre-LN layer normalization
        x_norm_self = self.norm_self(x)

        q_self, k_self, v_self = self.query(x_norm_self), self.key(x_norm_self), self.value(x_norm_self)
        attn_output_self, attn_weights_self = self.multi_head_self_attention(q_self, k_self, v_self, mask=mask)
        x = attn_output_self

        # pre-LN layer normalization
        x_norm_token = self.norm_token(x)
        token_norm = self.norm_token(self.learnable_token)

        q_token, k_token, v_token = \
            self.query_token(token_norm), self.key_token(x_norm_token), self.value_token(x_norm_token)
        attn_output_token, attn_weights_token = self.multi_head_token_attention(
            q_token, k_token, v_token, mask=mask_token
        )
        x = self.learnable_token + attn_output_token

        # pre-LN layer normalization
        x_norm_mlp = self.norm_mlp(x)
        x = x + self.mlp(x_norm_mlp)
        return x, [attn_weights_self, attn_weights_token]
