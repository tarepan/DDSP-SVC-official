import torch
from torch import nn
import math
from functools import partial
from einops import rearrange, repeat
import torch.nn.functional as F


class PCmer(nn.Module):
    """The encoder that is used in the Transformer model."""
    def __init__(self, num_layers: int, num_heads: int, dim_model: int, causal: bool = False):
        super().__init__()
        self.net = nn.Sequential(*[_EncoderLayer(num_heads, dim_model, causal) for _ in range(num_layers)])
    def forward(self, phone):
        return self.net(phone)


class _EncoderLayer(nn.Module):
    """Conformer encoder layer."""
    def __init__(self, n_head: int, ndim_model: int, causal: bool):
        """
        Args:
            parent - The encoder that the layers is created for.
        """
        super().__init__()

        # selfatt -> fastatt: performer!
        self.norm = nn.LayerNorm(ndim_model)
        self.attn = SelfAttention(dim=ndim_model, heads=n_head, causal=causal)
        self.local_mixer = ConformerConvModule(ndim_model, causal=causal)
        
    def forward(self, phone):
        # Res[LN-Attn]-Res[LocalMixer]
        phone = phone + (self.attn(self.norm(phone)))
        phone = phone + (self.local_mixer(phone))
        return phone 


#### ConvFF #########################################################################################################################
def calc_same_padding(kernel_size):
    """k=4 -> (2, 1)"""
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)


class Transpose(nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) == 2, 'dims must be a tuple of two dimensions'
        self.dims = dims

    def forward(self, x):
        return x.transpose(*self.dims)


class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)


class ConformerConvModule(nn.Module):
    """Alternative of Transformer's point-wise FF layer (LN-SegFC-GLU-DepthConv-SiLU-SegFC-Do)."""
    def __init__(self, dim, causal = False, expansion_factor = 2, kernel_size = 31, dropout = 0.):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Transpose((1, 2)),
            nn.Conv1d(      dim,       inner_dim * 2, 1),
            nn.GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim,     kernel_size, padding = padding),
            nn.SiLU(),
            nn.Conv1d(      inner_dim, dim,           1),
            Transpose((1, 2)),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
#### /ConvFF ########################################################################################################################


#### Attentions #####################################################################################################################

def linear_attention(q, k, v):
    if v is None:
        out = torch.einsum('...ed,...nd->...ne', k, q)
    else:
        k_cumsum = k.sum(dim = -2) 
        D_inv = 1. / (torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q)) + 1e-8)
        context = torch.einsum('...nd,...ne->...de', k, v)
        out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out


def orthogonal_matrix_chunk(cols, qr_uniform_q = False, device = None):
    unstructured_block = torch.randn((cols, cols), device = device)
    q, r = torch.linalg.qr(unstructured_block.cpu(), mode='reduced')
    q, r = map(lambda t: t.to(device), (q, r))

    # proposed by @Parskatt
    # to make sure Q is uniform https://arxiv.org/pdf/math-ph/0609050.pdf
    if qr_uniform_q:
        d = torch.diag(r, 0)
        q *= d.sign()
    return q.t()


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0, qr_uniform_q = False, device = None):
    nb_full_blocks = int(nb_rows / nb_columns)
    #print (nb_full_blocks)
    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q = qr_uniform_q, device = device)
        block_list.append(q)
    # block_list[n] is a orthogonal matrix ... (model_dim * model_dim)
    #print (block_list[0].size(), torch.einsum('...nd,...nd->...n', block_list[0], torch.roll(block_list[0],1,1)))
    #print (nb_rows, nb_full_blocks, nb_columns)
    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    #print (remaining_rows)
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q = qr_uniform_q, device = device)
        #print (q[:remaining_rows].size())
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)
    
    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device = device).norm(dim = 1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device = device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix


def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4):
    """
    Args:
        data
        projection_matrix
        is_query
        normalize_data
        eps
    """
    b, h, *_ = data.shape
    # (batch size, head, length, model_dim)

    # normalize model dim
    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    # what is ration?, projection_matrix.shape[0] --> 266
    
    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b = b, h = h)
    projection = projection.type_as(data)

    #data_dash = w^T x
    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    # diag_data = D**2 
    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)
    
    if is_query:
        data_dash = ratio * (torch.exp(data_dash - diag_data - torch.max(data_dash, dim=-1, keepdim=True).values) + eps)
    else:
        data_dash = ratio * (torch.exp(data_dash - diag_data                                               + eps)      )

    return data_dash.type_as(data)


class FastAttention(nn.Module):
    def __init__(self, dim_heads, causal = False):
        super().__init__()

        # Projection parameters
        nb_features = int(dim_heads * math.log(dim_heads))
        self.create_projection = partial(gaussian_orthogonal_random_matrix, nb_rows=nb_features, nb_columns=dim_heads, scaling=0, qr_uniform_q=False)
        self.register_buffer('projection_matrix', self.create_projection())

        # Attention function
        if causal:
            # optimized CUDA executor
            import fast_transformers.causal_product.causal_product_cuda
            causal_linear_fn = partial(causal_linear_attention)
        self.attn_fn = linear_attention if not causal else causal_linear_fn

    @torch.no_grad()
    def redraw_projection_matrix(self):
        projections = self.create_projection()
        self.projection_matrix.copy_(projections)
        del projections

    def forward(self, q, k, v):
        # # no projection - Q (queries) and K (keys) will be softmax-ed as in the original efficient attention paper
        #     q = q.softmax(dim = -1)
        #     k = torch.exp(k) if self.causal else k.softmax(dim = -2)
        # # generalized attention
        #     q = generalized_kernel(q, kernel_fn=nn.ReLU(), projection_matrix=self.projection_matrix,                 device=q.device)
        #     k = generalized_kernel(q, kernel_fn=nn.ReLU(), projection_matrix=self.projection_matrix,                 device=k.device)
        q = softmax_kernel(q, projection_matrix=self.projection_matrix, is_query=True)
        k = softmax_kernel(k, projection_matrix=self.projection_matrix, is_query=False)
        return self.attn_fn(q, k, v)


class SelfAttention(nn.Module):
    def __init__(self, dim, heads = 8, causal = False):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        self.heads = heads
        dim_head = 64
        inner_dim = dim_head * heads
        self.fast_attention = FastAttention(dim_head, causal=causal)

        self.to_q   = nn.Linear(dim,       inner_dim)
        self.to_k   = nn.Linear(dim,       inner_dim)
        self.to_v   = nn.Linear(dim,       inner_dim)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(0.)

    @torch.no_grad()
    def redraw_projection_matrix(self):
        self.fast_attention.redraw_projection_matrix()

    def forward(self, x):
        # Prepare QKV - SegFC + multi-head reshape
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        # Run attention
        out = self.fast_attention(q, k, v)

        # Reshape + SegFC
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return self.dropout(out)
#### /Attentions ####################################################################################################################
