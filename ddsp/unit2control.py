import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from .pcmer import PCmer


def split_to_dict(tensor, tensor_splits):
    """Split a tensor into a dictionary of multiple tensors."""
    labels = []
    sizes = []

    for k, v in tensor_splits.items():
        labels.append(k)
        sizes.append(v)

    tensors = torch.split(tensor, sizes, dim=-1)
    return dict(zip(labels, tensors))


class Unit2Control(nn.Module):
    def __init__(self, input_channel, n_spk: int, output_splits):
        """
        Args:
            input_channel - Feature dimension size of unit series
            n_spk         - The number of speakers
            output_splits - Output shape specifier
        """
        super().__init__()

        # PreNet - Conv-GN-LReLU-Conv
        kernel = 3
        self.stack = nn.Sequential(
                nn.Conv1d(input_channel, 256, kernel, padding="same"),
                nn.GroupNorm(4,          256),
                nn.LeakyReLU(),
                nn.Conv1d(256,           256, kernel, padding="same")) 

        # Embedding
        ndim_emb = 256
        ## fo/phase/volume continuous embedding :: (*, 1) -> (*, Emb)
        self.f0_embed     = nn.Linear(1, ndim_emb)
        self.phase_embed  = nn.Linear(1, ndim_emb)
        self.volume_embed = nn.Linear(1, ndim_emb)
        ## spk discrete embedding :: (*, 1) -> (*, Emb)
        self.spk_embed = nn.Embedding(n_spk, ndim_emb)

        # Conformer
        self.decoder = PCmer(num_layers=3, num_heads=8, dim_model=256, dim_keys=256, dim_values=256, residual_dropout=0.1, attention_dropout=0.1)
        self.norm = nn.LayerNorm(256)
        # PostNet - Linear
        self.n_out = sum([v for _, v in output_splits.items()])
        self.dense_out = weight_norm(nn.Linear(256, self.n_out))
        # Output split
        self.output_splits = output_splits

    def forward(self, units, f0, phase, volume, spk_id = None, spk_mix_dict = None):
        
        '''
        Args:
            units  :: (B, Frame, Feat) - Acoustic unit series
            f0     :: (..., 1)         - Fundamental tone's frequency contour
            phase  :: (..., 1)         -
            volume :: (..., 1)
            spk_id :: (..., 1)
            spk_mix_dict
        return: 
            dict of B x n_frames x feat
        '''

        # PreNet :: (B, Frame, Feat) -> (B, Feat, Frame) -> (B, Feat, Frame) -> (B, Frame, Feat)
        x = self.stack(units.transpose(1,2)).transpose(1,2)

        # Embedding
        ## Add continuous embeddings of fo/phase/volume to processed unit
        x = x + self.f0_embed((1+ f0 / 700).log()) + self.phase_embed(phase / np.pi) + self.volume_embed(volume)
        ## Add discrete or mixed discrete embeddings of spk to others
        if spk_mix_dict is not None:
            # Speaker mixing - weighted sum of each speaker's embeddings
            for k, v in spk_mix_dict.items():
                spk_id_tensor = torch.LongTensor(np.array([[k]])).to(units.device)
                x = x + v * self.spk_embed(spk_id_tensor - 1)
        else:
            # Speaker embedding
            x = x + self.spk_embed(spk_id - 1)

        # Conformer/PostNet
        x = self.decoder(x)
        x = self.norm(x)
        e = self.dense_out(x)

        return split_to_dict(e, self.output_splits) 
